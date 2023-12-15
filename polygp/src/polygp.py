from typing import List, Optional, Dict, Tuple, Union
from contextlib import redirect_stdout
from io import StringIO
import re

import optuna
from functools import partial

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import itertools
import more_itertools
from contextlib import redirect_stdout
from io import StringIO

from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Kernel
from sklearn import gaussian_process


import pandas as pd

import numpy as np
import math

import scipy.linalg as la

#from IPython.display import display

import logging # this is inherited from utils.py logging information

# set logging level 
LOGGING_LEVEL = 'DEBUG'
FORMAT = '%(asctime)s - POLYGP %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(LOGGING_LEVEL)

# Disable: DEBUG:matplotlib.font_manager:findfont: score(FontEntry(fname='C:\\Windows\\Fonts\\...
logging.getLogger('matplotlib.font_manager').disabled = True 

import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.ERROR)


def train_gp(train_df: pd.DataFrame,
             in_lst: List[str],
             out_lst: List[str],
             cat_lst: List[str],
             num_lst: List[str],
             max_evals: int,
             early_stopping_rounds: int,
             base_kernels: List[str],
             model_complexity: Optional[Dict[str, Union[int, str]]] = {
                 'max_n_prod': 3, 'max_n_sum': 5, 'comb_type': 'worepl'},
             study_init: Optional[optuna.study.Study] = None) -> Tuple[Pipeline, float, Kernel, optuna.study.Study]:        
    """Train a Gaussian Process regression model using hyperparameter optimization. The resulting model is actually a pipeline
    including data preprocessing

    Parameters:
    - train_df (pd.DataFrame): The training dataset.
    - in_lst (List[str]): List of input features.
    - out_lst (List[str]): List of output/target features.
    - cat_lst (List[str]): List of categorical features.
    - num_lst (List[str]): List of numerical features.
    - max_evals (int): Maximum number of evaluations for hyperparameter optimization.
    - early_stopping_rounds (int): Early stopping rounds for hyperparameter optimization.
    - base_kernels (List[str]): List of base kernels to consider.
    - model_complexity (dict): Dictionary specifying model complexity parameters.
        - 'max_n_prod' (int): Maximum number of product terms in the kernel.
        - 'max_n_sum' (int): Maximum number of terms in the kernel.
        - 'comb_type' (str): Combination type for the kernel, e.g., 'worepl' for with or without replacement.
    - study_init (optuna.study.Study or None): Optional pre-initialized Optuna study.

    Returns:
    - ppl_best (Pipeline): Trained pipeline including the Gaussian Process regression model.
    - best_loss: (float): loss of returned model pipeline
    - krnl_best: (Kernel): optimized kernel
    - study (optuna.study.Study): Optuna study containing optimization results.
    """
    logging.debug(f'train_polygp - Running kernel optimization')

    ### Update for 2D
    #base_kernels = [
    #        f'RBF(length_scale_bounds={ls[0]})',#f'RBF({[1]*len(in_lst)})',
    #        f'DotProduct(sigma_0_bounds={ls[0]})',
    #        f'Matern(length_scale_bounds={ls[0]})',#f'Matern({[1]*len(in_lst)})',
    #        f'RationalQuadratic(length_scale_bounds={ls[0]})',
    #        f'ExpSineSquared(length_scale_bounds={ls[0]}, periodicity_bounds={ls[0]})',
    #]

    # exponentiation 
    #base_kernels_exp = []
    #for bk in base_kernels:
    #    for ep in ['**2','**3']:
    #        base_kernels_exp.append(bk + ep)
    #    if bk in ['RBF()', 'Matern()',#f'RBF({[1]*len(in_lst)})', f'Matern({[1]*len(in_lst)})',
    #              'ExpSineSquared()', 
    #              'RationalQuadratic()']:
    #        base_kernels_exp.append(bk + '**0.5')
    #        base_kernels_exp.append(bk + '**1.5')
    #for bk in base_kernels: # exponentiation is typically used for DotProduct only
    #    if bk in ['DotProduct()']:
    #        for ep in ['**2']:
    #            base_kernels_exp.append(bk + ep)
    #base_kernels = base_kernels + base_kernels_exp

    # all polynomials
    polys = build_poly_kernel(base_kernels, 
                              model_complexity['max_n_prod'], 
                              model_complexity['max_n_sum'], 
                              'ConstantKernel(constant_value_bounds=(1e-5, 1e-1))*RBF(length_scale_bounds=(1e-5, 1e-1))+WhiteKernel(noise_level_bounds=(1e-5,1e-1))',
                              'ConstantKernel()',
                              )
    logging.debug(f"train_polygp - {len(polys)} available polynomials")

    max_retries = 10  # Maximum number of retries - when optimization fails
    study = None
    for _ in range(max_retries):
        try:
            study = run_optimization(max_evals=max_evals, 
                                     early_stopping_rounds=early_stopping_rounds,
                                     in_lst=in_lst, 
                                     out_lst=out_lst, cat_lst=cat_lst, 
                                     num_lst=num_lst, train_df=train_df, 
                                     polys=polys,
                                     study_init=study_init)

            #logging.debug(f"train_polygp - Best trial: {study.best_trial}")
            
            # If the operation is successful, break out of the loop
            break
        except Exception as e:
            # Handle the exception or simply log the error
            logging.error(f"train_polygp - Operation failed: {e}")
    else:
        # This block runs if the loop completes without a successful operation
        logging.debug("train_polygp - Maximum retries reached, operation failed.")

    # best trial and info stored with it
    best_trial = study.best_trial
    best_trial_info = best_trial.user_attrs.get('extra_info', {})
    # kernel of best trial and model pipeline
    krnl_best = best_trial_info['krnl']
    ppl_best = best_trial_info['mdl'] 
    # NOTE: this model may have a loss that is different than the best loss if an initial study is used.
    # In fact, the best loss from the list of reused trials (the one showed in progress bar as well)
    # was calculated with different conditions (example: different dataset size). Therefore, we recalculate the loss
    # here to reflect what was calculated during the itration in the optimization objective. 
    # Using best_loss = study.best_value reflects the best loss from the historical trails as well and is not reliable
    best_loss = _calculate_bic(len(train_df), ppl_best[-1].log_marginal_likelihood_value_, num_params = _calculate_num_params(krnl_best))

    logging.debug(f"train_polygp - Best kernel: {krnl_best}, {best_trial_info['loss']}")

    # train with constant kernel if fit fails
    if best_loss == 1e9: 
        logging.error('train_polygp - fit failed. Using Constant kernel')
        krnl_best = ConstantKernel(train_df[out_lst].mean())
        ppl_best = set_ppl(cat_lst, num_lst, krnl_best, is_optimizer=False)
        ppl_best.fit(train_df[in_lst],train_df[out_lst]) # this is needed to make sure we fit the whole pipeline
        best_loss = _calculate_bic(len(train_df), ppl_best[-1].log_marginal_likelihood_value_, num_params = _calculate_num_params(krnl_best)) #bic

    #### No need to train again, I saved ppl in extra_info - not sure if this will add extra time/memory in the long run
    # set pipeline and fit it with best params
    #ppl_best = set_ppl(cat_lst, num_lst, krnl_best, is_optimizer=False)
    #ppl_best.fit(train_df[in_lst],train_df[out_lst]) # this is needed to make sure we fit the whole pipeline
    #best_loss = calculate_bic(len(train_df), ppl_best[-1].log_marginal_likelihood_value_, num_params = calculate_num_params(krnl_best)) #bic

    return ppl_best, best_loss, krnl_best, study

def run_optimization(max_evals: int,
                     early_stopping_rounds: int,
                     in_lst: List[str],
                     out_lst: List[str],
                     cat_lst: List[str],
                     num_lst: List[str],
                     train_df: pd.DataFrame,
                     polys: List[str],
                     study_init: Optional[optuna.study.Study] = None) -> optuna.study.Study:
    """
    Run hyperparameter optimization.

    Parameters:
    - max_evals (int): Maximum number of evaluations for hyperparameter optimization.
    - early_stopping_rounds (int): Early stopping rounds for hyperparameter optimization.
    - in_lst (List[str]): List of input features.
    - out_lst (List[str]): List of output/target features.
    - cat_lst (List[str]): List of categorical features.
    - num_lst (List[str]): List of numerical features.
    - train_df (pd.DataFrame): The training dataset.
    - polys (List(str)): list of all possible polynomials
    - study_init (Optional[optuna.study.Study]): Optional pre-initialized Optuna study.

    Returns:
    - optuna.study.Study: Optuna study.
    """

    direction = 'minimize' # minimize llh
    #sampler = optuna.samplers.TPESampler(n_startup_trials=30) # set seed=123 for reproducibility
    sampler = optuna.samplers.CmaEsSampler(n_startup_trials=30, restart_strategy='ipop') 
    study = optuna.create_study(
        sampler=sampler,
        study_name='gpr_opti',
        direction=direction,
        #pruner=optuna.pruners.MedianPruner(
        #    n_startup_trials=3, n_warmup_steps=5, interval_steps=3
        #),
    )
    n_added = 0 # count added historical trials from initial study
    if study_init:
        # all the complete trials     
        all_trials = study_init.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        ## sort trials by their values in ascending order if minimizing
        sorted_trials = sorted(all_trials, key=lambda t: t.value, reverse=False if direction=='minimize' else True) 
        ## remove trials without kernel info
        sorted_trials = [i for i in sorted_trials if i.user_attrs]
        ## identify duplicates and remove
        sorted_trials_lst = [str(i.user_attrs['extra_info']['krnl']) for i in sorted_trials]
        logging.debug(f'run_optimization - {len(sorted_trials_lst)} trials with valid kernel available')  
        sorted_trials_dupl = {value: [index for index, element in enumerate(sorted_trials_lst) if element == value] for value in set(sorted_trials_lst) if sorted_trials_lst.count(value) > 1}
        sorted_trials_drop = []
        for _, indexes in sorted_trials_dupl.items():
            #print(f"Element {value} is duplicated at indexes: {indexes}")
            sorted_trials_drop.extend(indexes[1:])
        sorted_trials_drop = sorted(sorted_trials_drop)
        sorted_trials = [j for i, j in enumerate(sorted_trials) if i not in sorted_trials_drop]
        logging.debug(f'run_optimization - {len(sorted_trials)} unique trials')  
        # Get the top 30 trials
        top_trials = sorted_trials[:30]
        for trial in top_trials:
            study.add_trial(trial)
            n_added+=1
        logging.debug(f"run_optimization - top trial: {str(top_trials[0].user_attrs['extra_info']['krnl'])}")
        logging.debug(f'run_optimization - using previous study: {n_added} trials added. Total: {len(study.trials)}')  

    objective_part = partial(objective, in_lst=in_lst, out_lst=out_lst, 
                            cat_lst=cat_lst, num_lst=num_lst, 
                            train_df=train_df, polys=polys)

    study.optimize(func=objective_part, n_trials=max_evals+n_added, 
                callbacks=[partial(no_progress_loss, early_stopping_rounds=early_stopping_rounds+n_added)], # current trial number is set to zero when we create a new study
                show_progress_bar=True, gc_after_trial=False) #n_jobs=4

    return study

def objective(trial: optuna.Trial,
              in_lst: List[str],
              out_lst: List[str],
              cat_lst: List[str],
              num_lst: List[str],
              train_df: pd.DataFrame,
              polys: List[str]) -> float:
    """
    Objective function for hyperparameter optimization.

    Parameters:
    - trial (optuna.Trial): Optuna trial for hyperparameter tuning.
    - in_lst (List[str]): List of input features.
    - out_lst (List[str]): List of output/target features.
    - cat_lst (List[str]): List of categorical features.
    - num_lst (List[str]): List of numerical features.
    - train_df (pd.DataFrame): The training dataset.
    - polys (List(str)): list of all possible polynomials

    Returns:
    - float: Loss to be minimized during hyperparameter optimization.
    """

    # make sure the scales explored with the kernels are not too short and lead to overfitting
    #ls = []
    #for ii in in_lst:
    #    pdist = pairwise_distances(train_df[[ii]])
    #    pdist = pdist[np.tril_indices(pdist.shape[0], k=-1)] # exclude self-distances
    #    ls.append([np.min(pdist), np.max(pdist)])

    ## select number of base kernels to use
    poly_sel = trial.suggest_int('poly_sel', 0, len(polys)-1) 
    krnl = polys[poly_sel]        
    
    # https://stackoverflow.com/questions/25076883/creating-dynamically-named-variables-in-a-function-in-python-3-understanding-e
    # Create a StringIO object to capture the output 
    output_catcher = StringIO()

    exec_scope = {}
    code_to_exec = f'''from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, DotProduct, Matern, RationalQuadratic, ExpSineSquared
krnl = {krnl}
    '''
    try:
        with redirect_stdout(output_catcher):
            exec(code_to_exec, exec_scope)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    krnl = exec_scope['krnl']

    #print(krnl)

    # do not allow to have duplicate trials suggested
    for previous_trial in trial.study.trials:
        if previous_trial.state == optuna.trial.TrialState.COMPLETE and trial.params == previous_trial.params:
            logging.debug(f"objective - Duplicated trial: {trial.params}, return {previous_trial.value}")
            # do not return: allow for duplicates because different fits can lead to different solutions
            #return previous_trial.value 
            # use previously found kernel with optimized parameter values
            krnl = previous_trial.user_attrs['extra_info']['krnl'] 

    # trying model with give parameter values
    loss = 1e9
    ppl = set_ppl(cat_lst, num_lst, krnl, is_optimizer=True) #set pipeline
    llh = 1e9
    #y_std_pred = 1e9
    try:
        # subsample using stratification of target values
        # use a small number of data points to be fast
        if len(train_df)>50:
            X_train, _, y_train, _, _ = split_stratify(train_df, in_lst, out_lst, 
                                                        train_size=min(50, len(train_df)), 
                                                        n_bins=min(10, len(train_df)),
                                                        random_state=None)
            train_dfs = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
            # set pipeline and fit using llh
            ppl.fit(train_dfs[in_lst], train_dfs[out_lst])
            gpr_ppl_branch = ppl[-1]
            krnl = gpr_ppl_branch.kernel_
            num_params = _calculate_num_params(krnl)
            # refit on the whole data without optimizing the kernel hyperparameters using LLH
            ppl = set_ppl(cat_lst, num_lst, krnl, is_optimizer=False) 
            ppl.fit(train_df[in_lst], train_df[out_lst])
            gpr_ppl_branch = ppl[-1]
            llh = gpr_ppl_branch.log_marginal_likelihood_value_
            #llh = _calculate_bic(len(train_dfs), llh, num_params) #bic
            loss = _calculate_bic(len(train_df), llh, num_params) #bic

            #_, cov = ppl.predict(train_df, return_cov = True)     
            #alpha = abs(gpr_ppl_branch.alpha_.ravel())                        
            #loss = _calculate_bic2(len(train_df), llh, cov, alpha)
            # predict on the whole set
            #y_pred, _ = ppl.predict(train_df[in_lst], return_std=True)
            #loss = mean_squared_error(train_df[out_lst], y_pred)
        else:
            ppl.fit(train_df[in_lst], train_df[out_lst])
            gpr_ppl_branch = ppl[-1]#.named_steps['gaussianprocessregressor']
            krnl = gpr_ppl_branch.kernel_
            num_params = _calculate_num_params(krnl)
            llh = gpr_ppl_branch.log_marginal_likelihood_value_
            loss = _calculate_bic(len(train_df), llh, num_params) #bic

    except Exception as e:
        logging.error(f"objective - An error occurred: {e}")
        pass

    # add extra info to trial
    trial.set_user_attr('extra_info', {'krnl': krnl, 'mdl': ppl, 'loss': loss}) #

    return loss #+ np.mean(y_std_pred)

def build_composite_kernel(
    base_kernels: List,
    n_sel_kernels: int,
    comb_kern_idx: int,
    interc_coeff: str,
    coeff: str,
    n_prod: int,
    n_terms: int,
    comb_term_idx: int,
    mdl_complexity: dict
) -> str:
    """Builds a composite kernel based on specified parameters.

    Args:
        base_kernels (List): List of base kernels.
        n_sel_kernels (int): Number of base kernels to select.
        comb_kern_idx (int): Index of the base kernel combination to choose.
        interc_coeff (str): Intercept kernel.
        coeff (str): coefficient kernel.
        n_prod (int): Number of product terms.
        n_terms (int): Number of sum terms.
        comb_term_idx (int): Index of the sum combination term to choose.
        mdl_complexity (dict): Model complexity parameters.

    Returns:
        str: composite kernel component.
    """
    n_base_kernels = len(base_kernels)
    #logging.debug(f'build_terms: base kernels: {base_kernels}')

    assert n_sel_kernels <= n_base_kernels, 'You may not select more than base_kernels kernels'

    # given the base_kernels and how many base kernels we want to use n_sel_kernels,
    # we have len_comb_kernels combinations with n_base_kernels out of len(base_kernels) available base kernels
    max_n_base_kernels = math.comb(n_base_kernels,n_sel_kernels)
    assert comb_kern_idx <= max_n_base_kernels, 'You may not select an index greater than max_n_base_kernels combinations of base kernels'

    #logging.debug(f'build_terms: selected base kernels: {n_sel_kernels}; combination index: {comb_kern_idx}; max index: {max_n_base_kernels}')

    # jump to comb_term_sel - https://stackoverflow.com/questions/59043329/is-there-a-way-to-find-the-n%E1%B5%97%CA%B0-entry-in-itertools-combinations-without-convert
    comb_kern_sel = more_itertools.nth_combination(base_kernels, n_sel_kernels, comb_kern_idx-1)

    #logging.debug(f'build_terms: selected kernel combination: {comb_kern_sel}')

    # possible terms with interactions (product terms) from 1 to n_prod
    if mdl_complexity=='wrepl':
        max_n_sum_terms = sum(math.comb(n_sel_kernels + k - 1, k) for k in range(1, n_prod+1)) # same as len(prod_terms)
    else:
        max_n_sum_terms = sum(math.comb(n_sel_kernels, k) for k in range(1, n_prod+1)) # if prod_terms is without replacement

    #logging.debug(f'build_terms: max number of product terms with up to {n_prod} base kernels: {max_n_sum_terms}')
    #print(list(itertools.chain.from_iterable(itertools.combinations_with_replacement(comb_kern_sel, i) for i in range(1, n_prod+1))))

    max_n_sum_terms = math.comb(max_n_sum_terms, n_terms) 
    #logging.debug(f'build_terms: max number of combinations with {n_terms} sum terms from the avialble {max_n_sum_terms} of prod terms: {max_n_sum_terms}')
    assert comb_term_idx <= max_n_sum_terms, f'You may not select an index greater than {max_n_sum_terms} combinations of available sum terms'

    if mdl_complexity=='wrepl':
        prod_terms = list(itertools.chain.from_iterable(itertools.combinations_with_replacement(comb_kern_sel, i) for i in range(1, n_prod+1)))
    else:
        prod_terms = list(itertools.chain.from_iterable(itertools.combinations(comb_kern_sel, i) for i in range(1, n_prod+1))) # without replacement
    prod_terms = [coeff + '*' + ' * '.join(p_i) for p_i in prod_terms] 
    #logging.debug(f'build_terms: terms available to build final sum: {prod_terms}')

    # jump to comb_term_sel
    terms_sel = more_itertools.nth_combination(prod_terms, n_terms, comb_term_idx-1)

    return interc_coeff + ' + ' + ' + '.join(terms_sel)

def build_poly_kernel(
    base_kernels: List,
    interaction_degree: int,
    max_sum_terms: int,
    interc_coeff: str,
    coeff: str,
) -> List[str]:
    """Builds a composite kernel based on specified parameters.

    Args:
        base_kernels (List): List of base kernels.
        interaction_degree (int): Max degree of interaction. For example, 2: A*B, 3: A*B*C
        max_sum_terms (int): Max number of polynomial terms
        interc_coeff (str): Intercept kernel.
        coeff (str): coefficient kernel.
    Returns:
        str: composite kernel component.
    """

    poly_terms = [coeff + ' * ' + ' * '.join(i) for i in list(itertools.chain.from_iterable(itertools.combinations(base_kernels, i) for i in range(1, interaction_degree+1)))]
    polys = [interc_coeff + ' + ' + ' + '.join(i) for i in list(itertools.chain.from_iterable(itertools.combinations(poly_terms, i) for i in range(1, max_sum_terms+1)))]

    return polys


def split_stratify(
    data: pd.DataFrame,
    inputs: List[str],
    targets: List[str],
    n_bins: int = 10,
    train_size: float = 0.8,
    random_state: int = 123
) -> Tuple:
    """
    Stratify data based on numeric target.

    Args:
        data (DataFrame): The dataset.
        inputs (List[str]): List of input features.
        targets (List[str]): List of target variables (only 1 allowed for now).
        n_bins (int, optional): Number of bins for stratification. Defaults to 10.
        train_size (float, optional): Proportion of the dataset to include in the training split. Defaults to 0.8.
        random_state (int, optional): Seed for reproducibility. Defaults to 123.

    Returns:
        Tuple: Tuple containing the stratified input training set, input testing set, output training set, output testing set, the bin edges. 
    """

    assert len(targets)==1, "TODO: implement multitarget stratification"

    y0 = data[targets].values

    assert y0.dtype==float, "Target must be numeric"

    discr = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    strata = discr.fit_transform(y0).ravel().astype(str)

    X_train, X_test, y_train, y_test = train_test_split(data[inputs], data[targets], 
        stratify = strata, train_size = train_size, random_state=random_state)

    y0_edges = discr.bin_edges_[-1]

    return X_train, X_test, y_train, y_test, y0_edges

# can we calculate BIC using the actual DoF?
# https://rdrr.io/github/mattdneal/gaussianProcess/src/R/bic.R
def _calculate_bic(n, llh, num_params):
    bic = -2 * llh + num_params * np.log(n) #num_params*log(N) - 2*log(L)
    return bic

def _calculate_num_params(krnl):
    # kernel components
    num_params = 0
    for i in str(krnl).split(' + '):
        for j in i.split(' * '):
            if re.search('constant_value', j) or re.search(r'\*\*2', j): 
                num_params += 1
            if re.search('noise_level', j):
                num_params += 1
            if re.search('length_scale',j):
                num_params += 1
            if re.search('sigma_0',j):
                num_params += 1
            if re.search('alpha',j):
                num_params += 1
            if re.search('nu',j):
                num_params += 1
            if re.search('periodicity',j):
                num_params += 1
            #num_params += 1
    return num_params

def set_ppl(cat_lst, num_lst, krnl, is_optimizer, random_state=None):

    # preprocessing pipeline
    ppl_pre = []
    cat_proc = OneHotEncoder(handle_unknown="ignore", 
                            sparse=False, 
                            drop='if_binary') 
    num_proc = StandardScaler()#PowerTransformer('yeo-johnson', standardize=True)
    ppl_pre.append(
        ColumnTransformer(
            transformers=[
                ('one-hot-encoder', cat_proc, cat_lst), 
                ('scaler', num_proc, num_lst), 
            ],
            remainder='passthrough', # selection of inputs happen during fit anyway
        )
    )  

    # set params for GPR
    # no optimization: we're using hyperparameter tuning
    gpr = gaussian_process.GaussianProcessRegressor(
        optimizer=None if not is_optimizer else "fmin_l_bfgs_b", 
        n_restarts_optimizer = 10 if is_optimizer else 0,
        alpha=1e-10,
        normalize_y=False, 
        random_state = random_state,
        kernel=krnl,
    )

    #gpr = TransformedTargetRegressor(regressor = gpr, 
    #                        func = lambda x: np.log(x+1), #np.log(x+1) #x**(1.0/2.0)
    #                        inverse_func = lambda x: np.exp(x)-1) #np.exp(x)-1 #x**2

    # append to pipeline
    ppl_pre.append(gpr)      
    # create pipeline
    ppl = make_pipeline(*ppl_pre)

    return ppl

def no_progress_loss(study, trial, early_stopping_rounds=20):
    current_trial_number = trial.number
    if study.best_trials:#without this, you'll get no trials found yet error
        best_trial_number = study.best_trial.number
        should_stop = (current_trial_number - best_trial_number) >= early_stopping_rounds
        if should_stop:
            logging.debug("early stopping detected: %s", should_stop)
            study.stop()

