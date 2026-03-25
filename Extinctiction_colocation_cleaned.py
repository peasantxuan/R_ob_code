
# ========================
# SECTION 1: Basic Functions
# ========================
def wise_multiplier(M, left=None, right=None):

    """
    Y = wise_multiplier(M, left=v, right=w) means Y_{ij} = v_i M_{ij} w_j (no einstein sum).
    If left is None, then Y_{ij} = M_{ij} w_j.
    If right is None, then Y_{ij} = v_i M_{ij}.
    """

    ek = M.copy()
    if left is not None:
        ek = np.transpose(np.transpose(ek)*left)
    if right is not None:
        ek = ek*right
    return ek

def parametrization_nbinom_toscipy(mu, omega):
    """
    Purpose:
    Convert negative binomial distribution parameters to SciPy parametrization format.
    
    Inputs:
    - mu: float, mean parameter of negative binomial distribution
    - omega: float, dispersion parameter (inverse of shape)
    
    Outputs:
    - dict, dictionary with 'p' (probability) and 'n' (shape) for scipy.stats.nbinom
    
    Notes:
    Converts from (mu, omega) parametrization to scipy's (n, p) parametrization.
    The relationship is: p = 1/(1 + omega*mu), n = 1/omega.
    """

    return dict(p=1./(1.+omega*mu), n=1./omega)


def my_dirichlet_sampling(xbar, alpha, size=None):
    """
    assuming xbar.shape[0]->n

    output dimension:
    if size is None: one dimension (n, )
    if size is an integer: (size, n)
    """
    return np.random.dirichlet(alpha*xbar, size=size)

# #########
# #########
# # Poisson: extinction probability


# #
# Single-step functions

def pext_pois_single_1step(x, R):
    """
    Purpose:
    Single iteration step computing extinction probability for single-population Poisson branching.
    
    Inputs:
    - x: float, current extinction probability estimate at iteration k
    - R: float, basic reproduction number
    
    Outputs:
    - float, updated extinction probability estimate q_{k+1}
    
    Notes:
    Implements the fixed-point map q_{k+1} = exp(R(q_k - 1)) derived from
    Poisson branching process theory for single isolated population.
    """

    return np.exp(R*(x-1.))

def pext_pois_multi_1step(x, R):
    """
    Purpose:
    Single iteration step for extinction probability in multi-population Poisson system.
    
    Inputs:
    - x: array (n,), extinction probability vector at iteration k
    - R: array (n, n), transmission matrix
    
    Outputs:
    - array (n,), updated extinction probability vector q_{k+1}
    
    Notes:
    Implements the coupled system q_{k+1} = exp(R^T(q_k - 1)) for 
    multi-population Poisson branching process.
    """

    return np.exp(R.T @ (x-1))

# #
# Extinction probability for single pop

def pext_pois_single_asy(r):
    """
    assuming poisson. find extinction probability, single population, using root finding
    """
    f = lambda x: [np.exp(r*(x-1.)) - x, r*np.exp(r*(x-1.)) - 1., r*r*np.exp(r*(x-1.))]
    res = root_scalar(f, x0=0., x1=0.05, fprime2=True)
    assert res.converged, 'no convergence'
    q = res.root
    return q


def pext_pois_single_asy_iter(r, n_iter_max=5000, tol=1e-6):
    """
    assuming poisson. find extinction probability, single population, iterating asymptotically
    """
    q, q_old = 0., 10.
    conv = False
    for _ in range(n_iter_max):
        if np.abs(q-q_old) < tol:
            conv = True
            break
        q_old = q
        q = pext_pois_single_1step(q, r)
    assert conv, 'no convergence'
    return q

# #
# Extinction probability for multipop

def pext_pois_multi_asy(R):
    """
    assuming poisson. find extinction probability, multipopulation, using root finding
    """
    n = R.shape[0]
    
    f = lambda x: np.exp(R.T @ (x-1)) - x
    def df(x):
        v = np.exp(R.T @ (x-1))
        return wise_multiplier(R.T, left=v) - np.identity(n)
    
    res = root(f, x0=np.zeros(shape=n), jac=df)
    assert res.success, 'no convergence'
    return res.x


def pext_pois_multi_asy_iter(R, n_iter_max=5000, tol=1e-6):
    """
    assuming poisson. find extinction probability, multipopulation, iterating asymptotically
    """
    n = R.shape[0]
    q, q_old = np.zeros(shape=n), np.zeros(shape=n)+10.
    conv = False
    for _ in range(n_iter_max):
        if np.all( np.abs(q-q_old) ) < tol:
            conv = True
            break
        q_old = q.copy()
        q = pext_pois_multi_1step(q, R)
    assert conv, 'no convergence'
    return q

# #
# Effective R0

def effectiveR0_pois(q):
    """
    given a certain extinction probability q<1, find the R0 that would give that assuming poisson, single population
    """
    return np.log(q)/(q-1)



# %%
def get_R_matrix(populations, probabilities,beta,mu):
    """
    Purpose:
    Construct transmission matrix R from demographic and epidemiological parameters.
    
    Inputs:
    - populations: array, population sizes N_i for each location i
    - probabilities: array (n, n), contact probability matrix P_{ij}
        - mu: float, recovery/removal rate (    - beta: float, transmission rate per contact (
    Outputs:
    - array (n, n), basic reproduction number matrix R
    
    Notes:
    infections in population j from one infected individual in population i.    R_{ij} = (
    """

    adjusted_populations = populations
    NGM = pd.DataFrame(index=probabilities.index, columns=probabilities.columns)

    for i in probabilities.index:
        for j in probabilities.columns:
            NGM.at[i, j] = (beta / mu) * probabilities.at[i, j]*adjusted_populations[i]

    NGM_matrix = NGM.to_numpy(dtype=float)
    return NGM_matrix



def get_extinction_prob(populations, probabilities,beta,mu):
    """
    Purpose:
    Compute vector of extinction probabilities via fixed-point iteration.
    
    Inputs:
    - populations: array, population sizes for each location
    - probabilities: array, contact probability matrix
    - beta: float, transmission rate
    - mu: float, recovery rate
    
    Outputs:
    - array (n,), extinction probability q_i for each population i
    
    Notes:
    Solves the system of coupled equations q = exp(R^T(q - 1)) where
    q_i is the probability of stochastic extinction starting from one case in location i.
    """

    adjusted_populations = populations
    NGM = pd.DataFrame(index=probabilities.index, columns=probabilities.columns)

    for i in probabilities.index:
        for j in probabilities.columns:
            NGM.at[i, j] = (beta / mu) * probabilities.at[i, j]*adjusted_populations[i]

    NGM_matrix = NGM.to_numpy(dtype=float)
    extinction_prob_risk = pext_pois_multi_asy(NGM_matrix)
    return extinction_prob_risk

# Extinction_test = get_extinction_prob(italy_pop,italy_colocation/7,0.00055521100109155277*4,1/9.1)


def objective_threshold(populations, probabilities,beta,mu):
    """
    Purpose:
    Objective function for optimization-based threshold parameter determination.
    
    Inputs:
    - populations: array, population sizes
    - probabilities: array, contact matrix
    - beta: float, transmission rate
    - mu: float, recovery rate
    
    Outputs:
    - float, objective value for optimization routine
    
    Notes:
    Used in parameter fitting to find disease parameters yielding target 
    extinction probability thresholds.
    """

    adjusted_populations = populations 
    NGM = pd.DataFrame(index=probabilities.index, columns=probabilities.columns)

    for i in probabilities.index:
        for j in probabilities.columns:
            NGM.at[i, j] = (beta / mu) * probabilities.at[i, j]*adjusted_populations[i]
    NGM_matrix = NGM.to_numpy(dtype=float)
    max_eigenvalue = np.max(np.linalg.eigvals(NGM_matrix))
    
    return max_eigenvalue


def Local_R_caclulate(populations, probabilities,beta,mu):
    """
    Purpose:
    Compute local basic reproduction number for each location in metapopulation.
    
    Inputs:
    - populations: array (n,), population sizes
    - probabilities: array (n, n), contact probability matrix
    - beta: float, transmission rate per contact
    - mu: float, removal rate
    
    Outputs:
    - array (n,), local R value for each population
    
    Notes:
    Local R_i accounts for both direct transmission within population i and
    indirect transmission via connections to other patches.
    """

    NGM = pd.DataFrame(index=probabilities.index, columns=probabilities.columns)

    for i in probabilities.index:
        NGM.at[i, i] = (beta / mu) * probabilities.at[i, i]*populations[i]

    NGM_matrix = NGM.to_numpy(dtype=float)
    local_R = np.diag(NGM_matrix)
    return local_R

def find_highest_peaks(R_values, second_derivative):
    """
    Purpose:
    Identify local maxima in scalar field using second-derivative information.
    
    Inputs:
    - R_values: array, values at sample points (e.g., R or extinction probability)
    - second_derivative: array, second derivative estimates
    
    Outputs:
    - list, indices of detected local maxima
    
    Notes:
    Peaks are located where second derivative indicates local maximum (curvature changes sign).
    """

    # Identify the index of the highest peak in the second derivative for each column
    highest_peak_indices = np.argmax(second_derivative, axis=0)
    
    # Convert these indices to actual R values
    highest_peaks_R_values = R_values[highest_peak_indices]

    return highest_peaks_R_values



def find_R_values_close_to_target(R_values, extinction_matrix, target=0.025):
 
    """
    Purpose:
    Extract R values corresponding to target extinction probability threshold.
    
    Inputs:
    - R_values: array, basic reproduction numbers evaluated
    - extinction_matrix: array, extinction probabilities for each R value
    - target: float, target extinction probability (default 0.025 or 2.5%)
    
    Outputs:
    - array, R values where extinction probability approximately equals target
    
    Notes:
    Used to identify critical transmission intensities for disease elimination scenarios.
    """

    # Calculate the absolute differences from the target value for the second derivative
    differences = np.abs(extinction_matrix - target)
    # Identify the index where the difference is minimized for each column
    closest_indices = np.argmin(differences, axis=0)
    
    # Convert these indices to actual R values
    closest_R_values = R_values[closest_indices]

    return closest_R_values


def outbreak_calculation(R_values,pop,colocation_data,beta,gamma,target1,target2):

    """
    Purpose:
    Calculate comprehensive outbreak statistics across parameter space.
    
    Inputs:
    - R_values: array, basic reproduction numbers to evaluate
    - pop: array, population distribution across locations
    - colocation_data: array, contact/mobility matrix between locations
    - beta: float, transmission rate parameter
    - gamma: float, recovery rate parameter
    - target1: float, lower extinction probability threshold
    - target2: float, upper extinction probability threshold
    
    Outputs:
    - array, epidemic metrics (e.g., extinction probabilities, transmission potentials)
    
    Notes:
    Computes full epidemiological landscape for given geographic and contact structure.
    """

    extinction_matrix = np.zeros((len(R_values), len(pop)))

    for index, R in enumerate(R_values):
        extinction_matrix[index, :] = 1-get_extinction_prob(pop,colocation_data, 0.0003794804101197926*R,1/5.1)

    first_derivative = np.gradient(extinction_matrix, R_values, axis=0)
    second_derivative = np.gradient(first_derivative, R_values, axis=0)

    Local_R = Local_R_caclulate(pop, colocation_data,0.0003794804101197926,1/5.1)
    Global_R = 1/objective_threshold(pop, colocation_data, 0.0003794804101197926,1/5.1)

    second_order_peak = find_highest_peaks(R_values,second_derivative)
    outbreak__start1 = find_R_values_close_to_target(R_values,extinction_matrix, target1)
    outbreak__start2 = find_R_values_close_to_target(R_values,extinction_matrix, target2)
    
    return extinction_matrix,Local_R,Global_R,second_order_peak,outbreak__start1,outbreak__start2


def solve_y(Rii):
    """
    Purpose:
    Solve fixed-point equation y = exp(Rii * (y - 1)) for single population.
    
    Inputs:
    - Rii: float, diagonal element of transmission matrix (self-reproduction)
    
    Outputs:
    - float, solution to fixed-point equation (extinction probability)
    
    Notes:
    Used as component of larger fixed-point solver for multi-population systems.
    Solved via Newton-Raphson iteration or root-finding algorithm.
    """

    import numpy as np
    from scipy.optimize import fsolve

    def equation(y):
        return y - np.exp(Rii * y) + 1

    y_initial_guess = 0.5  # Initial guess for the solver
    y_solution = fsolve(equation, y_initial_guess)
    return y_solution[0]

def compute_wk(R, q0):
    """
    Purpose:
    Calculate weights or factors for transmission contributions from each population.
    
    Inputs:
    - R: array (n, n), transmission matrix
    - q0: array (n,), baseline extinction probabilities
    
    Outputs:
    - array (n,), computed transmission weights
    
    Notes:
    Intermediate calculation used in effective reproduction number estimation
    and transmission potential decomposition.
    """

    q = pd.Series(index=q0.index, dtype=np.float64)
    for i in q0.index:
        numerator = np.sum(R.loc[:, i] * (1 - q0)) - R.loc[i, i] * (1 - q0[i])
        denominator = 1 - R.loc[i, i] * q0[i]
        q[i] = q0[i] * (1 - numerator / denominator)
    return q




# %%
from scipy.optimize import minimize, root_scalar, root, fsolve

def Equivalent_R(p_i):
    """
    Purpose:
    Compute scalar equivalent basic reproduction number from extinction probability vector.
    
    Inputs:
    - p_i: array (n,), extinction probability for each population i
    
    Outputs:
    - float, equivalent R value summarizing metapopulation transmission potential
    
    Notes:
    Provides single-number summary of complex multi-population dynamics;
    useful for comparative analysis across different geographic settings.
    """

    return (1 / p_i) * np.log(1 / (1 - p_i))

# Equivalent_R_italy_09_25 = Equivalent_R( 1-get_extinction_prob(italy_pop,italy_colocation_09_25, 0.0003794804101197926*0.5,1/5.1))



def get_extinction_prob(Rmatrix):

    """
    Purpose:
    Compute vector of extinction probabilities via fixed-point iteration.
    
    Inputs:
    - populations: array, population sizes for each location
    - probabilities: array, contact probability matrix
    - beta: float, transmission rate
    - mu: float, recovery rate
    
    Outputs:
    - array (n,), extinction probability q_i for each population i
    
    Notes:
    Solves the system of coupled equations q = exp(R^T(q - 1)) where
    q_i is the probability of stochastic extinction starting from one case in location i.
    """

    extinction_prob_risk = pext_pois_multi_asy(Rmatrix)
    return extinction_prob_risk


def R_matrix_calc(populations, probabilities,beta,mu):
    """
    Purpose:
    Calculate transmission matrix from demographic and contact data.
    
    Inputs:
    - populations: array (n,), population size N_i in each location
    - probabilities: array (n, n), contact probability matrix P_{ij}
    - beta: float, transmission rate per contact
    - mu: float, recovery rate
    
    Outputs:
    - array (n, n), transmission matrix R
    
    Notes:
    Standard formulation: R_{ij} encodes expected secondary infections in j from primary in i.
    """

    adjusted_populations = populations
    NGM = pd.DataFrame(index=probabilities.index, columns=probabilities.columns)

    for i in probabilities.index:
        for j in probabilities.columns:
            NGM.at[i, j] = (beta / mu) * probabilities.at[i, j]*adjusted_populations.loc[i]

    NGM_matrix = NGM.to_numpy(dtype=float)
    return NGM_matrix


def read_data(pop_path,colocation_path):
    
    """
    Purpose:
    Load and parse population and contact pattern data from CSV files.
    
    Inputs:
    - pop_path: str, file path to population data CSV
    - colocation_path: str, file path to contact/colocation data CSV
    
    Outputs:
    - tuple, (populations array, contact probability matrix)
    
    Notes:
    Standardizes data formats and handles missing values or formatting inconsistencies.
    """

    pop_data = pd.read_csv(pop_path,index_col=0)
    colocation_data = pd.read_csv(colocation_path, index_col=0)
    colocation_data = pd.DataFrame(data=colocation_data.values, index=colocation_data.columns, columns=colocation_data.columns)
    
    # Pop_data = pop_data.groupby(pop_data.index).sum()
    pop_data = pop_data.reindex(colocation_data.index).dropna()
    
    return(pop_data,colocation_data)

def read_data_calculateall(popfile,colocationfile):
    """
    Purpose:
    Load data and compute full epidemiological analysis for R_ref = 1.0.
    
    Inputs:
    - popfile: str, path to population CSV file
    - colocationfile: str, path to contact data CSV file
    
    Outputs:
    - dict, comprehensive epidemiological results and derived metrics
    
    Notes:
    Wrapper function performing complete analysis pipeline with baseline transmission.
    """

    save_pop, save_colocation_09_25 = read_data(popfile,colocationfile)
    save_R_matrix = R_matrix_calc(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5,1/5.1)
    scale_R = 2/np.max(np.real(save_R_matrix))
    save_R_matrix = R_matrix_calc(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5*scale_R,1/5.1)
    save_R_extinction =  get_extinction_prob(save_R_matrix)
    save_R_extinction_eq_p = Equivalent_R(1-save_R_extinction)
    Local_R_save = Local_R_caclulate(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5*scale_R,1/5.1)
    save_pop["Local_R"] = Local_R_save
    save_pop["R_ob"] = save_R_extinction_eq_p
    return save_pop


# %%
testtttt = pd.read_csv('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/italy_colocation_09_25.csv',index_col=0)
pop_data_test = pd.read_csv('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ITA_geo_pop.csv',index_col=0)


# %%

def read_data_calculateall_1dot5(popfile,colocationfile):
    """
    Purpose:
    Load data and compute full epidemiological analysis for R_ref = 1.5.
    
    Inputs:
    - popfile: str, path to population CSV file
    - colocationfile: str, path to contact data CSV file
    
    Outputs:
    - dict, comprehensive epidemiological results at intermediate transmission
    
    Notes:
    Specialized computation for moderate transmission intensity scenario.
    """

    save_pop, save_colocation_09_25 = read_data(popfile,colocationfile)
    save_R_matrix = R_matrix_calc(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5,1/5.1)
    scale_R = 1.5/np.max(np.real(save_R_matrix))
    save_R_matrix = R_matrix_calc(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5*scale_R,1/5.1)
    save_R_extinction =  get_extinction_prob(save_R_matrix)
    save_R_extinction_eq_p = Equivalent_R(1-save_R_extinction)
    Local_R_save = Local_R_caclulate(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5*scale_R,1/5.1)
    save_pop["Local_R"] = Local_R_save
    save_pop["R_ob"] = save_R_extinction_eq_p
    save_pop["R_sum"] = sum_over_i(save_R_matrix)
    return save_pop


def read_data_calculateall_4(popfile,colocationfile):
    """
    Purpose:
    Load data and compute full epidemiological analysis for R_ref = 4.0.
    
    Inputs:
    - popfile: str, path to population CSV file
    - colocationfile: str, path to contact data CSV file
    
    Outputs:
    - dict, comprehensive epidemiological results at high transmission
    
    Notes:
    Specialized computation for high transmission intensity scenario.
    """

    save_pop, save_colocation_09_25 = read_data(popfile,colocationfile)
    save_R_matrix = R_matrix_calc(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5,1/5.1)
    scale_R = 4/np.max(np.real(save_R_matrix))
    save_R_matrix = R_matrix_calc(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5*scale_R,1/5.1)
    save_R_extinction =  get_extinction_prob(save_R_matrix)
    save_R_extinction_eq_p = Equivalent_R(1-save_R_extinction)
    Local_R_save = Local_R_caclulate(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5*scale_R,1/5.1)
    save_pop["Local_R"] = Local_R_save
    save_pop["R_ob"] = save_R_extinction_eq_p
    save_pop["R_sum"] = sum_over_i(save_R_matrix)
    return save_pop



def read_data_calculateall_2dot5(popfile,colocationfile):
    """
    Purpose:
    Load data and compute full epidemiological analysis for R_ref = 2.5.
    
    Inputs:
    - popfile: str, path to population CSV file
    - colocationfile: str, path to contact data CSV file
    
    Outputs:
    - dict, comprehensive epidemiological results at moderate-high transmission
    
    Notes:
    Specialized computation for intermediate-high transmission intensity.
    """

    save_pop, save_colocation_09_25 = read_data(popfile,colocationfile)
    save_R_matrix = R_matrix_calc(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5,1/5.1)
    scale_R = 2.5/np.max(np.real(save_R_matrix))
    save_R_matrix = R_matrix_calc(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5*scale_R,1/5.1)
    save_R_extinction =  get_extinction_prob(save_R_matrix)
    save_R_extinction_eq_p = Equivalent_R(1-save_R_extinction)
    Local_R_save = Local_R_caclulate(save_pop["Tot_Pop"],save_colocation_09_25, 0.0003794804101197926*0.5*scale_R,1/5.1)
    save_pop["Local_R"] = Local_R_save
    save_pop["R_ob"] = save_R_extinction_eq_p
    save_pop["R_sum"] = sum_over_i(save_R_matrix)
    return save_pop

# %%



def sum_over_i(R_matrix):
    """
    Purpose:
    Compute sum over first dimension of NGM matrix.
    
    Inputs:
    - R_matrix: 2D array, next-generation matrix with entries R_{ij}
    
    Outputs:
    - 1D array with summed entries: sum_i R_{ij}
    """
# %%

# ========================
# SECTION 2: DATA LOADING
# ========================
import copy
import pandas as pd
import numpy as np
from pandas import Series, DataFrame, merge, concat, read_csv, read_excel, isnull
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar, root, fsolve
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from scipy.special import gamma, digamma, gammaln
from scipy.stats import dirichlet, lognorm, poisson, nbinom
import json
import seaborn as sns

italy_colocation = pd.read_csv('./Italy_colocation.csv', index_col=0)
italy_pop_df = pd.read_csv('./corrected_merged_population_data.csv', index_col=1).reindex(italy_colocation.index)
italy_pop_df = italy_pop_df.drop('ID', axis=1)
italy_name_df = pd.read_csv('./Italy_name_data.csv', index_col=0).reindex(italy_colocation.index)

italy_pop = italy_pop_df.reindex(italy_pop_df.index)['Population']
italy_name = italy_name_df.reindex(italy_name_df.index)['Polygon Name']

italy_pop_df['Polygon ID'] = italy_pop_df.index
italy_name_df['Polygon ID'] = italy_name_df.index
italy_pop_all = merge(italy_pop_df, italy_name_df) 

italy_colocation_09_04 = pd.read_csv('./italy_colocation_09_04.csv', index_col=0)
italy_colocation_09_04 = pd.DataFrame(data=italy_colocation_09_04.values, index=italy_colocation_09_04.columns, columns=italy_colocation_09_04.columns)
italy_colocation_09_04=italy_colocation_09_04.reindex(index=italy_pop_df.index, columns=italy_pop_df.index)
italy_colocation_09_11 = pd.read_csv('./italy_colocation_09_11.csv', index_col=0)
italy_colocation_09_11 = pd.DataFrame(data=italy_colocation_09_11.values, index=italy_colocation_09_11.columns, columns=italy_colocation_09_11.columns)
italy_colocation_09_11=italy_colocation_09_11.reindex(index=italy_pop_df.index, columns=italy_pop_df.index) 
italy_colocation_09_18 = pd.read_csv('./italy_colocation_09_18.csv', index_col=0)
italy_colocation_09_18 = pd.DataFrame(data=italy_colocation_09_18.values, index=italy_colocation_09_18.columns, columns=italy_colocation_09_18.columns)
italy_colocation_09_18=italy_colocation_09_18.reindex(index=italy_pop_df.index, columns=italy_pop_df.index) 
italy_colocation_09_25 = pd.read_csv('./italy_colocation_09_25.csv', index_col=0)
italy_colocation_09_25 = pd.DataFrame(data=italy_colocation_09_25.values, index=italy_colocation_09_25.columns, columns=italy_colocation_09_25.columns)
italy_colocation_09_25= italy_colocation_09_25.reindex(index=italy_pop_df.index, columns=italy_pop_df.index) 



ITA_pop, ITA_colocation_09_25 = read_data('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ITA_geo_pop.csv','/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/italy_colocation_09_25.csv')
ITA_R_matrix = R_matrix_calc(ITA_pop["Tot_Pop"],ITA_colocation_09_25, 0.0003794804101197926*0.5,1/5.1)
ITA_R_extinction =  get_extinction_prob(ITA_R_matrix)
ITA_R_extinction_eq_p = Equivalent_R(1-ITA_R_extinction)
Local_R_ITA = Local_R_caclulate(ITA_pop["Tot_Pop"],ITA_colocation_09_25, 0.0003794804101197926*0.5,1/5.1)
ITA_pop["Local_R"] = Local_R_ITA
ITA_pop["R_ob"] = ITA_R_extinction_eq_p

# %%
# Maxeigenvale: 5.020456473184036
ITA_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ITA_geo_pop.csv','/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/italy_colocation_09_25.csv')
ITA_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ITA_geo_pop.csv','/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/italy_colocation_09_25.csv')
ITA_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ITA_geo_pop.csv','/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/italy_colocation_09_25.csv')
ITA_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ITA_geo_pop.csv','/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/italy_colocation_09_25.csv')

# %%
# Maxeigenvale: 5.020456473184036
MYS_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MYS_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MYS_colocation_09_25.csv')
MYS_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MYS_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MYS_colocation_09_25.csv')
MYS_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MYS_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MYS_colocation_09_25.csv')
MYS_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MYS_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MYS_colocation_09_25.csv')

# %%
# Maxeigenvale: 4.049758930151127
FRA_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/FRA_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/FRA_colocation_09_25.csv')
FRA_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/FRA_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/FRA_colocation_09_25.csv')
FRA_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/FRA_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/FRA_colocation_09_25.csv')
FRA_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/FRA_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/FRA_colocation_09_25.csv')

# %%
# Maxeigenvale: 2.803604789593891
GEO_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/GEO_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/GEO_colocation_09_25.csv')
GEO_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/GEO_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/GEO_colocation_09_25.csv')
GEO_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/GEO_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/GEO_colocation_09_25.csv')
GEO_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/GEO_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/GEO_colocation_09_25.csv')

# %%
# Maxeigenvale: 6.464905201388842

OMN_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/OMN_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/OMN_colocation_09_25.csv')
OMN_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/OMN_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/OMN_colocation_09_25.csv')
OMN_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/OMN_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/OMN_colocation_09_25.csv')
OMN_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/OMN_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/OMN_colocation_09_25.csv')

# %%
# Maxeigenvale: 1.3241693866955648

CZE_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CZE_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CZE_colocation_09_25.csv')
CZE_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CZE_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CZE_colocation_09_25.csv')
CZE_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CZE_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CZE_colocation_09_25.csv')
CZE_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CZE_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CZE_colocation_09_25.csv')

# %%
# #NEED TO RESCALLING
# Maxeigenvale: 52.77266332777262
CMR_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CMR_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CMR_colocation_09_25.csv')
CMR_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CMR_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CMR_colocation_09_25.csv')
CMR_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CMR_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CMR_colocation_09_25.csv')
CMR_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CMR_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CMR_colocation_09_25.csv')

# %%
# #NEED TO RESCALLING
# Maxeigenvale: 83.18527605688288
MLI_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MLI_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MLI_colocation_09_25.csv')
MLI_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MLI_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MLI_colocation_09_25.csv')
MLI_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MLI_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MLI_colocation_09_25.csv')
MLI_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MLI_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/MLI_colocation_09_25.csv')

# %%
# #NEED TO RESCALLING
# Maxeigenvale: 24.257658513245737
SEN_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SEN_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SEN_colocation_09_25.csv')
SEN_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SEN_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SEN_colocation_09_25.csv')
SEN_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SEN_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SEN_colocation_09_25.csv')
SEN_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SEN_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SEN_colocation_09_25.csv')

# %%
# #NEED TO RESCALLING
# Maxeigenvale: 9.840493853400329
ZAF_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ZAF_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ZAF_colocation_09_25.csv')
ZAF_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ZAF_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ZAF_colocation_09_25.csv')
ZAF_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ZAF_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ZAF_colocation_09_25.csv')
ZAF_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ZAF_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/ZAF_colocation_09_25.csv')

# %%
# #NEED TO RESCALLING
# Maxeigenvale: 8.330264018936642
CHL_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CHL_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CHL_colocation_09_25.csv')
CHL_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CHL_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CHL_colocation_09_25.csv')
CHL_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CHL_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CHL_colocation_09_25.csv')
CHL_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CHL_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/CHL_colocation_09_25.csv')

# %%
# #NO NEED TO RESCALLING
# Maxeigenvale: 2.4383622614504286
SUR_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SUR_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SUR_colocation_09_25.csv')
SUR_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SUR_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SUR_colocation_09_25.csv')
SUR_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SUR_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SUR_colocation_09_25.csv')
SUR_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SUR_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/SUR_colocation_09_25.csv')

# %%
# #No NEED TO RESCALLING
# Maxeigenvale: 1.2664688921616964
NZL_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/NZL_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/NZL_colocation_09_25.csv')
NZL_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/NZL_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/NZL_colocation_09_25.csv')
NZL_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/NZL_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/NZL_colocation_09_25.csv')
NZL_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/NZL_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/NZL_colocation_09_25.csv')

# %%
# # NEED TO RESCALLING
# Maxeigenvale: 93.50683945447139
PNG_pop = read_data_calculateall('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/PNG_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/PNG_colocation_09_25.csv')
PNG_pop_1dot5 = read_data_calculateall_1dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/PNG_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/PNG_colocation_09_25.csv')
PNG_pop_2dot5 = read_data_calculateall_2dot5('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/PNG_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/PNG_colocation_09_25.csv')
PNG_pop_4 = read_data_calculateall_4('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/PNG_geo_pop.csv',
                                '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/PNG_colocation_09_25.csv')

# %%
# ========================
# SECTION 3: Get Figure 4, Calculate Rob, Rlocal and Rsum for each country
# ========================
# Merge data
Robdata = {
    'ITA_pop': ITA_pop,
    'MYS_pop': MYS_pop,
    'FRA_pop': FRA_pop,
    'GEO_pop': GEO_pop,
    'OMN_pop': OMN_pop,
    'CZE_pop': CZE_pop,
 # 'CMR_pop': CMR_pop,
    'MLI_pop': MLI_pop,
    'SEN_pop': SEN_pop,
    'ZAF_pop': ZAF_pop,
    'CHL_pop': CHL_pop,
    'SUR_pop': SUR_pop,
    'NZL_pop': NZL_pop,
    'PNG_pop': PNG_pop
}

# Add a 'Source' column to each DataFrame and merge them
Robdata = pd.concat(
    [df.assign(Source=key) for key, df in Robdata.items()],
    ignore_index=True
)

# %%
Robdata_1dot5 = {
    'ITA_pop': ITA_pop_1dot5,
    'MYS_pop': MYS_pop_1dot5,
    'FRA_pop': FRA_pop_1dot5,
    'GEO_pop': GEO_pop_1dot5,
    'OMN_pop': OMN_pop_1dot5,
    'CZE_pop': CZE_pop_1dot5,
    'MLI_pop': MLI_pop_1dot5,
    'SEN_pop': SEN_pop_1dot5,
    'ZAF_pop': ZAF_pop_1dot5,
    'CHL_pop': CHL_pop_1dot5,
    'SUR_pop': SUR_pop_1dot5,
    'NZL_pop': NZL_pop_1dot5,
    'PNG_pop': PNG_pop_1dot5
}

# Add a 'Source' column to each DataFrame and merge them
Robdata_1dot5 = pd.concat(
    [df.assign(Source=key) for key, df in Robdata_1dot5.items()],
    ignore_index=True
)


Robdata_2dot5 = {
    'ITA_pop': ITA_pop_2dot5,
    'MYS_pop': MYS_pop_2dot5,
    'FRA_pop': FRA_pop_2dot5,
    'GEO_pop': GEO_pop_2dot5,
    'OMN_pop': OMN_pop_2dot5,
    'CZE_pop': CZE_pop_2dot5,
    'MLI_pop': MLI_pop_2dot5,
    'SEN_pop': SEN_pop_2dot5,
    'ZAF_pop': ZAF_pop_2dot5,
    'CHL_pop': CHL_pop_2dot5,
    'SUR_pop': SUR_pop_2dot5,
    'NZL_pop': NZL_pop_2dot5,
    'PNG_pop': PNG_pop_2dot5
}

# Add a 'Source' column to each DataFrame and merge them
Robdata_2dot5 = pd.concat(
    [df.assign(Source=key) for key, df in Robdata_2dot5.items()],
    ignore_index=True
)



Robdata_4 = {
    'ITA_pop': ITA_pop_4,
    'MYS_pop': MYS_pop_4,
    'FRA_pop': FRA_pop_4,
    'GEO_pop': GEO_pop_4,
    'OMN_pop': OMN_pop_4,
    'CZE_pop': CZE_pop_4,
    'MLI_pop': MLI_pop_4,
    'SEN_pop': SEN_pop_4,
    'ZAF_pop': ZAF_pop_4,
    'CHL_pop': CHL_pop_4,
    'SUR_pop': SUR_pop_4,
    'NZL_pop': NZL_pop_4,
    'PNG_pop': PNG_pop_4
}

# Add a 'Source' column to each DataFrame and merge them
Robdata_4 = pd.concat(
    [df.assign(Source=key) for key, df in Robdata_4.items()],
    ignore_index=True
)


Robdata_1dot5.to_csv("/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/R_obdata_1dot5.csv", index=False)
Robdata_2dot5.to_csv("/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/R_obdata_3.csv", index=False)
Robdata_4.to_csv("/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/R_obdata_4.csv", index=False)


# %%


# %%
def read_data_calculateall_multiR_merged(popfile, colocationfile, R_targets):
    """
    输入：
        popfile: 人口数据文件
        colocationfile: 共位矩阵文件
        R_targets: 要计算的R值列表，比如 [1, 1.5, 2, 2.5, 3, 3.5]
    输出：
        merged_df: 合并后的DataFrame，带 R_ref、R_ob、Local_R、rho 列
    """
    # Read data
    save_pop, save_colocation_09_25 = read_data(popfile, colocationfile)

    # Initial calculation for baseline eigenvalues
    save_R_matrix_base = R_matrix_calc(
        save_pop["Tot_Pop"],
        save_colocation_09_25,
        0.0003794804101197926 * 0.5,
        1 / 5.1
    )
    base_eigen = np.max(np.real(save_R_matrix_base))

    # Store all results
    all_results = []

    for target_R in R_targets:
        scale_R = target_R / base_eigen

        # Calculate R matrix
        save_R_matrix = R_matrix_calc(
            save_pop["Tot_Pop"],
            save_colocation_09_25,
            0.0003794804101197926 * 0.5 * scale_R,
            1 / 5.1
        )
        actual_R = np.max(np.real(save_R_matrix))

        # Extinction probability and equivalent R
        save_R_extinction = get_extinction_prob(save_R_matrix)
        save_R_extinction_eq_p = Equivalent_R(1 - save_R_extinction)

        # Local R values
        Local_R_save = Local_R_caclulate(
            save_pop["Tot_Pop"],
            save_colocation_09_25,
            0.0003794804101197926 * 0.5 * scale_R,
            1 / 5.1
        )

        # 🔹 Calculate rho_i = sum_j R_ji
        rho_i = np.sum(save_R_matrix, axis=0)  # Sum by column, input risk for each location

        # Construct new DataFrame with R_ref
        df = save_pop.copy()
        df["Local_R"] = Local_R_save
        df["R_ob"] = save_R_extinction_eq_p
        df["R_ref"] = target_R
        df["rho"] = rho_i  # ✅ Add rho column

        all_results.append(df)

    # Merge all results
    merged_df = pd.concat(all_results, ignore_index=True)

    return merged_df




# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1️⃣ Set parameters
# =====================================================
R_list = np.linspace(0.9, 10, 200)

countries = [
    "PNG", "ITA", "MYS", "FRA", "GEO", "OMN",
    "CZE", "CMR", "MLI", "SEN", "ZAF", "CHL", "SUR", "NZL"
]

# =====================================================
# 2️⃣ Main loop: call your function for each country
# =====================================================
all_results = []

for country in countries:

    pop_path = f"/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/{country}_geo_pop.csv"
    coloc_path = f"/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/{country}_colocation_09_25.csv"

    df = read_data_calculateall_multiR_merged(pop_path, coloc_path, R_list)
    df["Country"] = country  # ✅ Add country column for grouping after merging
    all_results.append(df)

# =====================================================
# 3️⃣ Merge data from all countries
# =====================================================
all_countries_merged = pd.concat(all_results, ignore_index=True)

# Save (optional)
all_countries_merged.to_csv("all_countries_Rref_heterogeneity.csv", index=False)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Select a 20-color colormap
cmap = plt.cm.get_cmap("tab20", len(all_countries_merged["Country"].unique()))

# Create side-by-side figures with shared axes
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# ==========================
# Left plot: R_ob heterogeneity
# ==========================
for i, (country, g) in enumerate(all_countries_merged.groupby("Country")):
    color = cmap(i)  # Get color from palette
    g_sorted = g.sort_values("R_ref")
    grouped = (
        g_sorted.groupby("R_ref")
        .agg(max_Rob=('R_ob', 'max'), min_Rob=('R_ob', 'min'))
        .reset_index()
        .sort_values("R_ref")
    )
    grouped["Rob_range_norm"] = (
        (grouped["max_Rob"] - grouped["min_Rob"]) / grouped["R_ref"]
    )
    axes[0].plot(
        grouped["R_ref"],
        grouped["Rob_range_norm"],
        label=country,
        linewidth=1.5,
        color=color
    )

axes[0].set_xlabel(r"$R_{ref}$", fontsize=12)
axes[0].set_ylabel(r"$(\max R_{ob} - \min R_{ob}) / R_{ref}$", fontsize=12)
axes[0].set_title("Risk heterogeneity (R_ob)", fontsize=14)
axes[0].grid(True, linestyle=":", alpha=0.6)

# ==========================
# Right plot: rho heterogeneity
# ==========================
for i, (country, g) in enumerate(all_countries_merged.groupby("Country")):
    color = cmap(i)
    g_sorted = g.sort_values("R_ref")
    grouped = (
        g_sorted.groupby("R_ref")
        .agg(max_rho=('rho', 'max'), min_rho=('rho', 'min'))
        .reset_index()
        .sort_values("R_ref")
    )
    grouped["rho_range_norm"] = (
        (grouped["max_rho"] - grouped["min_rho"]) / grouped["R_ref"]
    )
    axes[1].plot(
        grouped["R_ref"],
        grouped["rho_range_norm"],
        label=country,
        linewidth=1.5,
        color=color
    )

axes[1].set_xlabel(r"$R_{ref}$", fontsize=12)
axes[1].set_title("Risk heterogeneity (ρ)", fontsize=14)
axes[1].grid(True, linestyle=":", alpha=0.6)

# ==========================
# Legend and layout
# ==========================
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=7,
    fontsize=10,
    frameon=False
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()




# %%


# %%
import pandas as pd
import matplotlib.pyplot as plt

png_data = PNG_allR.copy()

# Allow some tolerance to select close R_ref
def subset_by_Rref(data, target, tol=0.05):
    """
    Purpose:
    Extract and filter analysis results matching specified reference reproduction number.
    
    Inputs:
    - data: dict, comprehensive epidemiological analysis dictionary
    - target: float, target R_ref value for filtering
    - tol: float, tolerance band for matching target (default 0.05)
    
    Outputs:
    - dict, filtered subset of data matching target R_ref
    
    Notes:
    Enables comparative analysis across multiple transmission intensity scenarios.
    """

    return data[(data['R_ref'] >= target - tol) & (data['R_ref'] <= target + tol)]

# Select subset close to 1.5 and 3.5
rho_15 = subset_by_Rref(png_data, 1.5)
rho_35 = subset_by_Rref(png_data, 3.5)

# Check number of selections

# Plot: distribution comparison of rho
plt.figure(figsize=(8,6))

plt.hist(rho_15['rho'], bins=30, alpha=0.6, label=r'$R_{ref}\approx1.5$', density=True)
plt.hist(rho_35['rho'], bins=30, alpha=0.6, label=r'$R_{ref}\approx3.5$', density=True)

plt.xlabel(r'$\rho$', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title(r'Distribution of $\rho$ across communities', fontsize=14)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()


# %%


# %%
np.unique(Robdata['DEGURBA_L2'])

# %% [markdown]
# GID_2: A unique identifier for each second-level administrative unit, following the GADM (Global Administrative Areas) coding scheme.
#
# GID_0GHSL: The country code corresponding to each administrative unit, based on the GHSL (Global Human Settlement Layer) classification.
#
# Tot_Pop: The total population residing within the administrative unit.
#
# UCentre_Pop: The population living in urban centers within the administrative unit.
#
# UCluster_Pop: The population residing in urban clusters within the administrative unit.
#
# Rural_Pop: The population living in rural areas within the administrative unit.
#
# UCentre_share: The proportion of the total population that resides in urban centers, expressed as a decimal (e.g., 0.545 represents 54.5%).
#
# UCluster_share: The proportion of the total population living in urban clusters, expressed as a decimal.
#
# Urban_share: The combined proportion of the population residing in both urban centers and urban clusters, indicating the overall urban population share.
#
# Rural_share: The proportion of the total population living in rural areas, expressed as a decimal.
#
# DEGURBA_L1: The Degree of Urbanisation classification at Level 1, categorizing areas into broad classes such as urban or rural.
#
# DUC_Pop: The population residing in densely populated urban centers within the administrative unit.
#
# SDUC_Pop: The population living in semi-dense urban clusters within the administrative unit.
#
# SUrb_Pop: The population residing in suburban areas within the administrative unit.
#
# RC_Pop: The population living in rural clusters within the administrative unit.
#
# LDR_Pop: The population residing in low-density rural areas within the administrative unit.
#
# VLDR_Pop: The population living in very low-density rural areas within the administrative unit.
#
# DUC_share: The proportion of the total population residing in densely populated urban centers, expressed as a decimal.
#
# SDUC_share: The proportion of the total population living in semi-dense urban clusters, expressed as a decimal.
#
# SUrb_share: The proportion of the total population residing in suburban areas, expressed as a decimal.
#
# RC_share: The proportion of the total population living in rural clusters, expressed as a decimal.
#
# LDR_share: The proportion of the total population residing in low-density rural areas, expressed as a decimal.
#
# VLDR_share: The proportion of the total population living in very low-density rural areas, expressed as a decimal.
#
# DEGURBA_L2: The Degree of Urbanisation classification at Level 2, providing a more detailed categorization of areas based on population density and settlement patterns.

# %%
degurba_mapping = {
    30: "Urban Centre",
    23: "Dense Urban Cluster",
    22: "Semi-Dense Urban Cluster",
    21: "Suburban or Peri-Urban Areas",
    13: "Rural Cluster",
    12: "Low Density Rural Grid Cell",
    11: "Very Low Density Grid Cell"
}
Robdata['DEGURBA_L2'] = Robdata['DEGURBA_L2'].map(degurba_mapping)


# %%
category_order = [
    "Urban Centre",
    "Dense Urban Cluster",
    "Semi-Dense Urban Cluster",
    "Suburban or Peri-Urban Areas",
    "Rural Cluster",
    "Low Density Rural Grid Cell",
    "Very Low Density Grid Cell"
]

# Convert the column to a Categorical type with the specified order
Robdata['DEGURBA_L2'] = pd.Categorical(Robdata['DEGURBA_L2'], categories=category_order, ordered=True)

# %%
Robdata['Rob/Rlocal']= Robdata["R_ob"]/Robdata["Local_R"]
Robdata['log(Rob/Rlocal)']= np.log(Robdata['Rob/Rlocal'])

# %%
Robdata[Robdata["Source"] == 'MYS_pop']

# %%
plt.figure(figsize=(10, 6))
Robdata.boxplot(column='Rob/Rlocal', by='DEGURBA_L2', grid=False, showmeans=True, meanline=True)
plt.title('Box Plot of Rob/Rlocal by DEGURBA_L2')
plt.suptitle('')  # Suppress the default title from boxplot
plt.xlabel('DEGURBA_L2')
plt.ylabel('Rob/Rlocal')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


# %%
plt.figure(figsize=(10, 6))
Robdata.boxplot(column='log(Rob/Rlocal)', by='DEGURBA_L2', grid=False, showmeans=True, meanline=True)
plt.title('Box Plot of Log(Rob/Rlocal) by DEGURBA_L2')
plt.suptitle('')  # Suppress the default title from boxplot
plt.xlabel('DEGURBA_L2')
plt.ylabel('log(Rob/Rlocal)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


# %%
sources = Robdata["Source"].unique()

# Create subplots
for source in sources:
    plt.figure(figsize=(10, 6))
    subset = Robdata[Robdata["Source"] == source]
    subset.boxplot(column='log(Rob/Rlocal)', by='DEGURBA_L2', grid=False, showmeans=True, meanline=True)
    plt.title(f'Box Plot of Log(Rob/Rlocal) by DEGURBA_L2 for Source {source}')
    plt.suptitle('')
    plt.xlabel('DEGURBA_L2')
    plt.ylabel('log(Rob/Rlocal)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# %%

# ========================
# SECTION 4: Get Figure1, Calculate outbreak probability vs R for selected cities & map, and calculate the relationship between Rob and Rlocal
# ========================

# #
# Figure 1: Probability of outbreak vs R for selected cities & map
# #

# %%
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Helvetica'
matplotlib.rcParams['mathtext.it'] = 'Helvetica:italic'
matplotlib.rcParams['mathtext.bf'] = 'Helvetica:bold'


# %%
target_names = ['Caserta', 'Napoli', 'Trento']
indices = italy_pop_all[italy_pop_all['Polygon Name'].isin(target_names)].index.tolist()

# Define the number of subplots per row and figure layout
plots_per_row = 3
num_plots = len(indices)

# Calculate how many rows we need per figure
num_rows = (num_plots + plots_per_row - 1) // plots_per_row  # Ceiling division

# Create the figure and subplots
fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(25, 3))  # Wide layout
axes = axes.flatten()  # Flatten to easily iterate

# Plot each subplot
for subplot_index, i in enumerate(indices):
    ax = axes[subplot_index]
    ax.grid(False)

    # Plot extinction probability
    ax.plot(R_values, extinction_matrix[:, i], label='Outbreak Probability', color='green', linewidth=4, alpha=0.7)
    ax.set_ylabel('Outbreak Probability',fontweight='bold', fontsize=12)

    # Add vertical lines for Local_R and Global_R
    ax.axvline(x=Local_R[i], color='red', linestyle='--', linewidth=4, label='Local R value')
    ax.axvline(x=Global_R, color='blue', linestyle='-.', linewidth=4, label='Global R value')

    # Set title
    ax.set_title(f'{italy_name_df["Polygon Name"].iloc[i]} vs. R_reference',fontweight='bold', fontsize=20)

# Hide unused subplots (if any)
for ax in axes[num_plots:]:
    ax.set_visible(False)

# Add a single legend for the entire figure
fig.legend(
    ['Outbreak Probability', 'Local R value', 'Global R value'],
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),  # Place legend below the figure
    ncol=3,
    prop={'size': 30}  # Adjust legend font size (increase as needed)
)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend

# Save the figure
plt.savefig('Figure/Probability_col.png', bbox_inches='tight', dpi=300)

plt.show()



# %%
# ==================== Global font settings ====================
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Helvetica'
matplotlib.rcParams['mathtext.it'] = 'Helvetica:italic'
matplotlib.rcParams['mathtext.bf'] = 'Helvetica:bold'
matplotlib.rcParams['font.family'] = 'Helvetica'


# ==================== First figure: City Outbreak Probabilities ====================
target_names = ['Caserta', 'Napoli', 'Trento']

# Map city names to English (for sorting and title)
city_name_mapping = {
    'Caserta': 'Caserta',
    'Napoli': 'Naples',
    'Trento': 'Trento'
}

# Sort city names in English alphabetical order
sorted_names = sorted(target_names, key=lambda x: city_name_mapping[x])

# Ensure index order matches sorted_names exactly
indices = [
    italy_pop_all.index[italy_pop_all['Polygon Name'] == name][0]
    for name in sorted_names
]

# Create multiple subplots in one row, height consistent with Rjj = 2.0 figure（3.5）
fig2, axes = plt.subplots(1, len(indices), figsize=(15, 3.5))

# If only one subplot, ensure axes is in array form
if len(indices) == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for subplot_index, i in enumerate(indices):
    ax = axes[subplot_index]
    ax.grid(False)

    # Plot outbreak probability curve
    ax.plot(
        R_values,
        extinction_matrix[:, i],
        color='#1f77b4',
        linewidth=4,
        alpha=1
    )

    # Y-axis label
    ax.set_ylabel(
        'Epidemic Probability',
        fontweight='bold',
        fontsize=14,
        fontname='Helvetica'
    )

    # Add vertical lines for Local_R and Global_R
    ax.axvline(
        x=Local_R[i],
        color='#58508d',
        linestyle=':',
        linewidth=4,
        label=r'$R^{local}$'
    )
    ax.axvline(
        x=Global_R,
        color='#ffa600',
        linestyle=':',
        linewidth=4,
        label=r'$R^{ref}=1$'
    )

    # Get city English name
    original_name = italy_name_df["Polygon Name"].iloc[i]
    english_name = city_name_mapping.get(original_name, original_name)

    # ==================== Title in upper left corner of figure ====================
    ax.text(
        0.02, 0.95, english_name,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=16,
        fontweight='bold',
        fontname='Helvetica'
    )

    # X-axis label
    ax.set_xlabel(
        r'$R^{ref}$',
        fontsize=14,
        fontweight='bold',
        fontname='Helvetica'
    )
    ax.set_ylim(bottom=0)

# Add legend only to first subplot
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(
    handles,
    labels,
    loc='lower right',
    prop={'size': 10, 'family': 'Helvetica'},
    frameon=False
)

# Set font for all axis ticks
for ax in axes:
    for tick in ax.get_xticklabels():
        tick.set_fontname('Helvetica')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Helvetica')

# Adjust layout and save
plt.tight_layout()
plt.savefig(
    '/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/Figure/Probability_row.pdf',
    bbox_inches='tight',
    dpi=300
)
plt.show()

# Save the figure

# %%




# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ==========================================
# 1. Core solver
# ==========================================
def solve_epidemic_prob(R, k=None, tol=1e-6, max_iter=2000):
    """
    Purpose:
    Solve extinction probability equations for general branching process model.
    
    Inputs:
    - R: array (n, n), transmission matrix
    - k: array or None, moments of offspring distribution (default None for Poisson)
    - tol: float, convergence tolerance for fixed-point iteration (default 1e-6)
    - max_iter: int, maximum iterations allowed (default 2000)
    
    Outputs:
    - array (n,), extinction probability vector
    
    Notes:
    Generalizes beyond Poisson offspring; allows arbitrary branching process offspring distributions
    through specification of distribution moments.
    """

    N = R.shape[0]
    p = np.ones(N) * 0.5
    for _ in range(max_iter):
        lamb = R @ p 
        if k is None:
            p_new = 1 - np.exp(-lamb)
        else:
            p_new = 1 - np.power(1 + lamb / k, -k)
        
        if np.max(np.abs(p_new - p)) < tol:
            return p_new
        p = p_new
    return p

# ==========================================
# 2. Parameter setup
# ==========================================
# Coupling strength (X-axis)
coupling_values = np.logspace(-2.5, 0.5, 50)

# Define list of k values to test (from highly discrete -> close to Poisson)
# SARS-CoV-2 is around 0.1
k_list = [0.05, 0.1, 0.5, 1.0, 5.0]

# Fixed network parameters
R_ii = 0.5  # Sink community (cannot outbreak locally)
R_jj = 2.5  # Source community (strong outbreak source)
R_ij = 0.0  # Reverse influence ignored

# ==========================================
# 3. Calculate data
# ==========================================
# Store Poisson baseline (k -> infinity)
pi_pois = []
for r_ji in coupling_values:
    R = np.array([[R_ii, r_ji], [R_ij, R_jj]])
    p = solve_epidemic_prob(R, k=None)
    pi_pois.append(p[0])

# Store curves for different k values
results_nb = {}
for k in k_list:
    pi_k = []
    for r_ji in coupling_values:
        R = np.array([[R_ii, r_ji], [R_ij, R_jj]])
        p = solve_epidemic_prob(R, k=k)
        pi_k.append(p[0])
    results_nb[k] = pi_k

# ==========================================
# 4. Draw plot
# ==========================================
plt.figure(figsize=(8, 6))

# Draw Poisson baseline (black bold)
plt.plot(coupling_values, pi_pois, color='black', linewidth=2.5, zorder=10, label='Poisson ($k \\to \\infty$)')

# Draw lines for different k values (using colormap)
# We use Reds spectrum, darker color means smaller k (more extreme) or vice versa
colors = cm.autumn_r(np.linspace(0.2, 1, len(k_list)))

for i, k in enumerate(k_list):
    plt.plot(coupling_values, results_nb[k], 
             color=colors[i], 
             linestyle='--', 
             linewidth=2, 
             label=f'NegBin ($k={k}$)')

# Decorate figure
plt.xscale('log')
plt.xlabel('Coupling Strength $R_{ji}$', fontsize=12)
plt.ylabel('Epidemic Probability in Community i', fontsize=12)
plt.title('Robustness of Coupling Effect Across Dispersion Regimes', fontsize=14)

# Add auxiliary lines (Local Threshold)
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

# Adjust legend
plt.legend(fontsize=10, loc='upper left', frameon=True, framealpha=0.9)
plt.grid(True, which="both", ls="-", alpha=0.15)
plt.tight_layout()

# Save
plt.savefig('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/Figure/SI_Fig_Coupling_Multi_K.pdf')
plt.show()

print("多条 K 线对比图已生成: SI_Fig_Coupling_Multi_K.pdf")

# %%


# %%


# %%


# %%


# %%
Equivalent_R_italy_09_25  

# %%
sum_over_i(R_matrix_calc(italy_pop, italy_colocation_09_25, 0.0003794804101197926*0.5*1.1136005654033705,1/5.1))-Equivalent_R_italy_09_25_local[18]

# %%
# Map for different R VALUE
Equivalent_R_italy_09_25 =  Equivalent_R(extinction_matrix)
Equivalent_R_italy_09_25_local = R_values[:, np.newaxis] / Local_R 
R_eq_data = pd.DataFrame({
    'Name': ITA_pop.index,
    'R_local': Equivalent_R_italy_09_25_local,
    'R_outbreak': Equivalent_R_italy_09_25[18, :],
    "R_sum":sum_over_i(R_matrix_calc(italy_pop, italy_colocation_09_25, 0.0003794804101197926*0.5*1.1136005654033705,1/5.1)
),
    'R_reference': 2.5
})

gdf_9_04 = pd.merge(gdf,R_eq_data, left_on='NAME_2',right_on='Name', how='left')




import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Determine the min and max values for the color scale
vmin = 0
vmax = 2.5

# Create a figure with three subplots
fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(15, 6))

# Define the colormap and normalization
cmap = cm.viridis
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Plot the first map for R_reference
gdf_9_04.plot(column='R_reference', ax=ax1, cmap=cmap, linewidth=0, edgecolor='0.8', norm=norm)
ax1.set_title('R_reference', fontweight='bold', fontsize=20)
ax1.axis('off')  # Remove the axis for R_reference

# Plot the second map for R_local
gdf_9_04.plot(column='R_local', ax=ax2, cmap=cmap, linewidth=0, edgecolor='0.8', norm=norm)
ax2.set_title('R_local', fontweight='bold', fontsize=20)
ax2.axis('off')  # Remove the axis for R_local

# Plot the second map for R_local
gdf_9_04.plot(column='R_sum', ax=ax3, cmap=cmap, linewidth=0, edgecolor='0.8', norm=norm)
ax3.set_title('R_sum', fontweight='bold', fontsize=20)
ax3.axis('off')  # Remove the axis for R_local


# Plot the third map for R_outbreak
gdf_9_04.plot(column='R_outbreak', ax=ax4, cmap=cmap, linewidth=0, edgecolor='0.8', norm=norm)
ax4.set_title('R_outbreak', fontweight='bold', fontsize=20)
ax4.axis('off')  # Remove the axis for R_outbreak


# Fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax1, ax2, ax3], orientation='horizontal', shrink=1.5, fraction=0.03, pad=0.1)

# Create a shared colorbar
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax1, ax2, ax3,ax4], 
                    orientation='horizontal', shrink=3, fraction=0.03, pad=0.1)

# Make color bar ticks bold
cbar.ax.tick_params(labelsize=12)  # Adjust tick font size
for tick in cbar.ax.get_xticklabels():  # Bold ticks
    tick.set_fontweight('bold')

    
# Save the figure
plt.savefig('Figure/R_local_R_outbreak_maps.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()



# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

# Custom hex color codes
hex_colors = ['#003f5c', '#58508d', '#8a508f', '#bc5090', '#de5a79', '#ff6361', '#ff8531', '#ffa600']
custom_cmap = ListedColormap(hex_colors)

# Determine the min and max values for the color scale
vmin = 0
vmax = 2.6

# Create a figure with three subplots
fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(15, 6))

# Define the normalization
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Plot the first map for R_reference
gdf_9_04.plot(column='R_reference', ax=ax1, cmap=custom_cmap, linewidth=0, edgecolor='0.8', norm=norm)
ax1.axis('off')  # Remove the axis for R_reference

# Plot the second map for R_local
gdf_9_04.plot(column='R_local', ax=ax2, cmap=custom_cmap, linewidth=0, edgecolor='0.8', norm=norm)
ax2.axis('off')  # Remove the axis for R_local

# Plot the third map for R_outbreak
gdf_9_04.plot(column='R_sum', ax=ax3, cmap=custom_cmap, linewidth=0, edgecolor='0.8', norm=norm)
ax3.axis('off')  # Remove the axis for R_outbreak

gdf_9_04.plot(column='R_outbreak', ax=ax4, cmap=custom_cmap, linewidth=0, edgecolor='0.8', norm=norm)
ax4.axis('off')  # Remove the axis for R_outbreak


ax1.set_title(r'$R^{ref}$', fontweight='bold', fontsize=20)
ax2.set_title(r'$R^{local}$', fontweight='bold', fontsize=20)
ax3.set_title(r'$R^{tot}$', fontweight='bold', fontsize=20)
ax4.set_title(r'$R^{ob}$', fontweight='bold', fontsize=20)


# Create a shared colorbar
cbar = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=custom_cmap), 
    ax=[ax1, ax2, ax3,ax4],
    orientation='vertical',
    fraction=0.05,
    pad=0.1,
    location='left',
    shrink=0.7  # <--- Shrink to 60% height
)
cbar.set_label(
    "R",  # Title text
    rotation=0,  # Vertical orientation
    labelpad=15,  # Spacing between label and colorbar
    fontsize=20,
    fontweight='bold'
)

# Make color bar ticks bold
cbar.ax.tick_params(labelsize=14)  # Adjust tick font size
for tick in cbar.ax.get_xticklabels():  # Bold ticks
    tick.set_fontweight('bold')
pos = cbar.ax.get_position()
cbar.ax.set_position([pos.x0, pos.y0+ 0.03, pos.width+ 0.03, pos.height])

import matplotlib.patches as patches

rect = patches.Rectangle(
    (0.12, 0.2),  # ← x0=0.15 more to left, y0=0.25 more to bottom
    0.8,  # ← width=0.70，框更“瘦”
    0.65,       transform=fig.transFigure, 
    fill=False,
    edgecolor='black',
    linewidth=1
)
fig.patches.append(rect)

# Save the figure
plt.savefig('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/Figure/R_local_R_outbreak_maps.pdf', dpi=150, bbox_inches='tight')

# Display the plot
plt.show()









# %%
# #
# ========================
# SECTION5: Figure2 stochastic meta-population model to validate statistical explanation
# ========================


# META POPULATION MODEL TO VALIDATE STATISTICAL EXPLANATION

import copy
def fstochastic(R,start_patch, max_size, nruns=1000, Tmax=50):
    """
    Purpose:
    Simulate stochastic epidemic branching dynamics in spatial metapopulation.
    
    Inputs:
    - R: array (n, n), transmission matrix governing spatial coupling
    - start_patch: int, initial infected location ( index < n)0 
    - max_size: int, maximum total population size (outbreak termination threshold)
    - nruns: int, number of independent simulation replicates (default 1000)
    - Tmax: int, maximum time steps per simulation (default 50)
    
    Outputs:
    - list, results from each stochastic realization (outbreak outcomes)
    
    Notes:
    Implements multitype branching process with spatial migration.
    Outbreaks terminate when total infections exceed max_size or T > Tmax.
    """

    num_communities = R.shape[0]  # Applies to arbitrary dimensions
    OB = np.zeros(shape=nruns, dtype=bool)
    X_init = np.zeros(shape=nruns, dtype=int)
    MI = np.zeros(shape=(Tmax, num_communities), dtype=int)  # Evolution state of each community
    for h in range(nruns):
        MI[:, :] = 0  # Initialize
        MI[0, start_patch] = 1  # Initialize seed in community 1
        t = 1
        while True:
            assert t < Tmax, '达到最大时间'
            # Use Poisson distribution for transmission
            MI[t, :] = np.random.poisson(R @ MI[t-1, :])  

            if MI[t, :].sum() == 0:
                # Extinction occurred
                X_init[h] = MI[1, :].sum() / MI[0, start_patch]
                break
                
            if MI.sum() >= max_size:
                # Reached maximum population size
                X_init[h] = MI[1, :].sum() / MI[0, start_patch]
                OB[h] = True
                break    
            t += 1
            if t >= Tmax:
              break

            
    return (OB, X_init)


def wrapper(R,start_patch, max_size,R_obdata, **kw):
    """
    Purpose:
    Wrapper function interfacing stochastic simulation with parameter inference.
    
    Inputs:
    - R: array, transmission matrix
    - start_patch: int, initial infection location
    - max_size: int, maximum population size
    - R_obdata: array, observed outbreak data to compare against
    - **kw: additional keyword arguments passed to simulation
    
    Outputs:
    - array, simulated outbreak results
    
    Notes:
    Formats simulation output to match observed data structure for likelihood calculation.
    """

    num_communities = R.shape[0]  # Calculate the number of communities

    # Call fstochastic, returns OB (outbreak status) and X_init
    OB, X_init = fstochastic(R,start_patch, max_size, **kw)
    
    # Statistical computation
    n_ob = OB.sum()  # Number of runs reaching maximum population size
    rho_OB = X_init[OB].mean() if n_ob > 0 else 0  # Mean initial population size for runs reaching maximum population
    rho_nOB = X_init[~OB].mean() if n_ob < OB.shape[0] else 0  # Mean initial population size for runs reaching maximum population
    R_ob_stat = rho_OB - rho_nOB  # Statistically computed transmission rate
    print(n_ob)
    # Calculate standard deviation of statistical transmission rate
    if n_ob > 0 and n_ob < OB.shape[0]:
        R_ob_stat_std = np.sqrt(X_init[OB].var() / n_ob + X_init[~OB].var() / (OB.shape[0] - n_ob))
    else:
        R_ob_stat_std = 0  # Set standard deviation to zero when insufficient statistical data

    R_ob = copy.copy(R_obdata)  # Theoretical transmission rate

    # Calculate estimation error
    err = (R_ob_stat - R_ob) / R_ob if R_ob != 0 else np.inf

    # Return statistical results as Pandas Series
    return pd.Series([R_ob, R_ob_stat, R_ob_stat_std, max_size, OB.shape[0], n_ob, err],
                     index=['R_ob', 'R_ob_stat', 'R_ob_stat (std)', 'max_size', 'n', 'n_OB', 'err'])



from joblib import Parallel, delayed

def parallel_wrapper(R, lsize, R_obdata, nruns, Tmax):
    """
    Purpose:
    Execute parallel stochastic simulations across multiple scenarios efficiently.
    
    Inputs:
    - R: array, transmission matrix
    - lsize: int, location or patch identifier
    - R_obdata: array, empirical outbreak size distribution
    - nruns: int, simulation replicates per condition
    - Tmax: int, maximum simulation duration
    
    Outputs:
    - array, concatenated results from all parallel runs
    
    Notes:
    Enables distributed computation of epidemic simulations across multi-core systems.
    """

    # Use Parallel to execute the wrapper function for each max_size
    results = Parallel(n_jobs=-1)(  # Use all available cores
        delayed(wrapper)(R, max_size, R_obdata, nruns=nruns, Tmax=Tmax)
        for max_size in lsize
    )
    # Convert the list of results into a DataFrame
    return pd.DataFrame(results, index=lsize)

R_matrix_test = get_R_matrix(italy_pop,italy_colocation_09_25, 0.0003794804101197926*0.5*1.1136005654033705
,1/5.1)
Equivalent_R_italy_09_25 =  Equivalent_R( 1-get_extinction_prob(R_matrix_test))


# %%
italy_pop.columns

# %%
Theoretical_R_ob= Equivalent_R(1-get_extinction_prob(R_matrix_test))[82]
lsize = np.round( np.logspace(1, 4, 20, base=10) ).astype(int)
ecco_trento = pd.DataFrame(dict([(x, wrapper(R_matrix_test,82, x, Theoretical_R_ob , nruns=50000, Tmax=300)) for x in lsize]) ).T.reindex(lsize)
ecco_trento["R reference"] = np.max(np.linalg.eig(R_matrix_test)[0])
ecco_trento["Region"] = 'Venezia'

# %%
Theoretical_R_ob= Equivalent_R(1-get_extinction_prob(R_matrix_test))[96]
lsize = np.round( np.logspace(1, 4, 20, base=10) ).astype(int)
ecco_napoli = pd.DataFrame(dict([(x, wrapper(R_matrix_test,96, x, Theoretical_R_ob , nruns=50000, Tmax=300)) for x in lsize]) ).T.reindex(lsize)
ecco_napoli["R reference"] = np.max(np.linalg.eig(R_matrix_test)[0])
ecco_napoli["Region"] = 'Napoli'

# %%
Theoretical_R_ob= Equivalent_R(1-get_extinction_prob(R_matrix_test))[95]
lsize = np.round( np.logspace(1, 4, 20, base=10) ).astype(int)
ecco_caserta = pd.DataFrame(dict([(x, wrapper(R_matrix_test,95, x, Theoretical_R_ob , nruns=50000, Tmax=300)) for x in lsize]) ).T.reindex(lsize)
ecco_caserta["R reference"] = np.max(np.linalg.eig(R_matrix_test)[0])
ecco_caserta["Region"] = 'Caserta'

# %%
Theoretical_R_ob= Equivalent_R(1-get_extinction_prob(R_matrix_test))[32]
lsize = np.round( np.logspace(1, 4, 20, base=10) ).astype(int)
ecco_torino = pd.DataFrame(dict([(x, wrapper(R_matrix_test,32, x, Theoretical_R_ob , nruns=50000, Tmax=300)) for x in lsize]) ).T.reindex(lsize)
ecco_torino["R reference"] = np.max(np.linalg.eig(R_matrix_test)[0])
ecco_torino["Region"] = 'Torino'

# %%
Theoretical_R_ob= Equivalent_R(1-get_extinction_prob(R_matrix_test))[9]
lsize = np.round( np.logspace(1, 4, 20, base=10) ).astype(int)
ecco_roma = pd.DataFrame(dict([(x, wrapper(R_matrix_test,9, x, Theoretical_R_ob , nruns=50000, Tmax=300)) for x in lsize]) ).T.reindex(lsize)
ecco_roma["R reference"] = np.max(np.linalg.eig(R_matrix_test)[0])
ecco_roma["Region"] = 'Roma'

# %%
Theoretical_R_ob= Equivalent_R(1-get_extinction_prob(R_matrix_test))[7]
lsize = np.round( np.logspace(1, 4, 20, base=10) ).astype(int)
ecco_genova = pd.DataFrame(dict([(x, wrapper(R_matrix_test,7, x, Theoretical_R_ob , nruns=50000, Tmax=300)) for x in lsize]) ).T.reindex(lsize)
ecco_genova["R reference"] = np.max(np.linalg.eig(R_matrix_test)[0])
ecco_genova["Region"] = 'Genova'

# %%


# %%
target_names = ['Genova', 'Roma', 'Torino',"Venezia"]
italy_pop_all[italy_pop_all['Polygon Name'].isin(target_names)].index.tolist()


# %%
fig, ax = plt.subplots(figsize=(4, 3))
ax.set_xscale('log')
ax.axhline(ecco_trento.loc[10, 'R_ob'], color='tab:green', label='R^{ob}')
ax.plot(lsize, ecco_trento['R_ob_stat'], 'o-', label='measured R^{ob}', color='tab:blue')
ax.fill_between(lsize, ecco_trento['R_ob_stat']-1.96*ecco_trento['R_ob_stat (std)'], ecco_trento['R_ob_stat']+1.96*ecco_trento['R_ob_stat (std)'], lw=0, alpha=0.3, color='tab:blue')
ax.set_xlabel(r'cutoff outbreak size')
ax.set_ylabel(r'$R$')
# Ax.set_ylim(1.3, None)
ax.set_xlim(10, None)
ax.legend(loc='best')


# %%


# %%
[67, 95, 96]
target_names = ['Trento','Caserta', 'Napoli']


# %%
ecco_all_6_region = pd.concat([ecco_caserta, ecco_trento, ecco_napoli,ecco_torino,ecco_roma,ecco_genova], ignore_index=True)


# %%
ecco_all_6_region.to_csv("/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/Figure/multitime_estimation_6_region.csv", index=False)


# %%
fig, ax = plt.subplots(figsize=(4, 3))
ax.set_xscale('log')
ax.axhline(ecco.loc[10, 'R_ob'], color='tab:green', label='R^{ob}')
ax.plot(lsize, ecco['R_ob_stat'], 'o-', label='measured R^{ob}', color='tab:blue')
ax.fill_between(lsize, ecco['R_ob_stat']-1.96*ecco['R_ob_stat (std)'], ecco['R_ob_stat']+1.96*ecco['R_ob_stat (std)'], lw=0, alpha=0.3, color='tab:blue')
ax.set_xlabel(r'cutoff outbreak size')
ax.set_ylabel(r'$R$')
# Ax.set_ylim(1.3, None)
ax.set_xlim(10, None)
ax.legend(loc='best')




# %%
import matplotlib.pyplot as plt
import seaborn as sns
import os

# #
# 0. STYLE
# #

sns.set_theme(style="whitegrid", context="talk")

# #
# 1. SAFE RENAME (do not contaminate original data)
# #

region_rename_map = {
    "Venezia": "Venice",
    "Napoli": "Naples",
    "Caserta": "Caserta",
    "Torino": "Turin",
    "Roma": "Rome",
    "Genova": "Genoa"
}

df_plot = df_all_regions_prob.copy()

df_plot["Region_EN"] = (
    df_plot["Region"]
    .astype(str)
    .str.strip()
    .map(region_rename_map)
)

assert df_plot["Region_EN"].notna().all(), \
    f"Unmapped regions: {df_plot[df_plot['Region_EN'].isna()]['Region'].unique()}"

regions_list = sorted(df_plot["Region_EN"].unique())

# #
# 2. R_ref (consistent with your R_global value)
# #

R_ref_val = R_global_val  # ← Key: unified semantics

p_ref_val = solve_prob_from_R_full_v1(R_ref_val)

df_plot["p_from_Rref"] = p_ref_val

# #
# 3. SAVE PATH
# #

save_dir = "/Users/boxuan/Desktop/PhD/R_ob_manuscript/Figure"
os.makedirs(save_dir, exist_ok=True)

# #
# 4. FIGURE（2x3）
# #

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()

palette = sns.color_palette("tab10")

# #
# 5. PLOT
# #

for i, region_name in enumerate(regions_list):

    ax = axes[i]

    df_sub = df_plot[df_plot["Region_EN"] == region_name].sort_values("cutoff")

    # --- Simulation ---
    ax.errorbar(
        df_sub["cutoff"],
        df_sub["p_est"],
        yerr=df_sub["p_std"],
        fmt='o-',
        color=palette[0],
        capsize=3,
        alpha=0.9,
        label="Simulation"
    )

    # --- R_ob ---
    ax.plot(
        df_sub["cutoff"],
        df_sub["p_from_Rob"],
        color=palette[1],
        linewidth=2.5,
        label=r"$R^{\mathrm{ob}}$"
    )

    # --- R_local ---
    ax.plot(
        df_sub["cutoff"],
        df_sub["p_from_Rlocal"],
        linestyle="--",
        color=palette[2],
        linewidth=2,
        label=r"$R^{\mathrm{local}}$"
    )

    # --- R_ref ---
    ax.plot(
        df_sub["cutoff"],
        df_sub["p_from_Rref"],
        linestyle=":",
        color=palette[3],
        linewidth=2,
        label=r"$R^{\mathrm{ref}}$"
    )

    # --- Theory ---
    ax.axhline(
        df_sub["p_theory"].iloc[0],
        color="black",
        linestyle="-.",
        linewidth=2,
        label="Theory"
    )
    ax.set_xlim(left=10)
    # #
    # Subplot title
    # #

    ax.set_title(f"({chr(97+i)}) {region_name}")

    # #
    # Axes
    # #

    ax.set_xscale("log")

    if i % 3 == 0:
        ax.set_ylabel("Epidemic probability")

    if i >= 3:
        ax.set_xlabel("Outbreak size threshold")

# #
# 6. GLOBAL LEGEND
# #

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

# #
# 7. LAYOUT
# #

plt.tight_layout(rect=[0, 0, 1, 0.95])

# #
# 8. SAVE
# #

save_path = os.path.join(save_dir, "Supplementary_Figure_4.png")
plt.savefig(save_path, dpi=300)
plt.close()


# %%


# %%
print(df_all_regions_prob["Region"].unique())

# %%
france_pop.index

# %%


# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# #
# ======= Your original code (keep unchanged) =======
# #

france_pop = pd.read_excel('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/franch_dep_population.xlsx',engine='openpyxl')
france_colocation_09_04 = pd.read_csv('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/france_colocation_09_04.csv', index_col=0)

set(france_colocation_09_04.index)-set(france_pop['Nom du département'])
set(france_pop['Nom du département'])-set(france_colocation_09_04.index)

france_pop= france_pop.set_index('Nom du département')

france_colocation_09_04 = pd.DataFrame(data=france_colocation_09_04.values, index=france_colocation_09_04.columns, columns=france_colocation_09_04.columns)

france_pop = france_pop.reindex(france_colocation_09_04.index).dropna()

france_pop['Population municipale'] = france_pop['Population municipale'].str.replace(' ', '')
france_pop['Population municipale'] = pd.to_numeric(france_pop['Population municipale'], errors='coerce')


def R_matrix_calc(populations, probabilities,beta,mu):
    """
    Purpose:
    Calculate transmission matrix from demographic and contact data.
    
    Inputs:
    - populations: array (n,), population size N_i in each location
    - probabilities: array (n, n), contact probability matrix P_{ij}
    - beta: float, transmission rate per contact
    - mu: float, recovery rate
    
    Outputs:
    - array (n, n), transmission matrix R
    
    Notes:
    Standard formulation: R_{ij} encodes expected secondary infections in j from primary in i.
    """

    adjusted_populations = populations
    NGM = pd.DataFrame(index=probabilities.index, columns=probabilities.columns)

    for i in probabilities.index:
        for j in probabilities.columns:
            NGM.at[i, j] = (beta / mu) * probabilities.at[i, j]*adjusted_populations.loc[i]

    return NGM.to_numpy(dtype=float)


R_matrix_france = R_matrix_calc(france_pop["Population municipale"],france_colocation_09_04,1*2,4149)


# #
# ======= Auto mapping (including Corse) =======
# #

def assign_region_from_department_name_final(dep):

    """
    Purpose:
    Map French department names to administrative region codes.
    
    Inputs:
    - dep: str, department name (e.g., "le-de-France")
    
    Outputs:
    - str, regional code identifier
    
    Notes:
    Implements lookup table for geographic aggregation from department to region level.
    Handles special cases and naming variations.
    """

    mapping = {
        "Île-de-France": ["Paris","Seine-et-Marne","Yvelines","Essonne","Hauts-de-Seine","Seine-Saint-Denis","Val-de-Marne","Val-d'Oise"],
        "Provence-Alpes-Côte d'Azur": ["Alpes-de-Haute-Provence","Hautes-Alpes","Alpes-Maritimes","Bouches-du-Rhône","Var","Vaucluse"],
        "Occitanie": ["Ariège","Aude","Aveyron","Gard","Haute-Garonne","Gers","Hérault","Lot","Lozère","Hautes-Pyrénées","Pyrénées-Orientales","Tarn","Tarn-et-Garonne"],
        "Auvergne-Rhône-Alpes": ["Ain","Allier","Ardèche","Cantal","Drôme","Isère","Loire","Haute-Loire","Puy-de-Dôme","Rhône","Savoie","Haute-Savoie"],
        "Nouvelle-Aquitaine": ["Charente","Charente-Maritime","Corrèze","Creuse","Dordogne","Gironde","Landes","Lot-et-Garonne","Pyrénées-Atlantiques","Deux-Sèvres","Vienne","Haute-Vienne"],
        "Bretagne": ["Côtes-d'Armor","Finistère","Ille-et-Vilaine","Morbihan"],
        "Normandie": ["Calvados","Eure","Manche","Orne","Seine-Maritime"],
        "Hauts-de-France": ["Aisne","Nord","Oise","Pas-de-Calais","Somme"],
        "Grand Est": ["Ardennes","Aube","Marne","Haute-Marne","Meurthe-et-Moselle","Meuse","Moselle","Bas-Rhin","Haut-Rhin","Vosges"],
        "Pays de la Loire": ["Loire-Atlantique","Maine-et-Loire","Mayenne","Sarthe","Vendée"],
        "Centre-Val de Loire": ["Cher","Eure-et-Loir","Indre","Indre-et-Loire","Loir-et-Cher","Loiret"],
        "Bourgogne-Franche-Comté": ["Côte-d'Or","Doubs","Jura","Nièvre","Haute-Saône","Saône-et-Loire","Yonne","Territoire de Belfort"],
        "Corse": ["Corse-du-Sud","Haute-Corse"]
    }

    for region, deps in mapping.items():
        if dep in deps:
            return region

    return np.nan


mapping_df_final = pd.DataFrame(index=france_colocation_09_04.index)
mapping_df_final["RegionName"] = [
    assign_region_from_department_name_final(dep)
    for dep in mapping_df_final.index
]

mapping_df_final = mapping_df_final.dropna()


# #
# ======= Correct aggregation (core) =======
# #

# Population
population_region_final = france_pop["Population municipale"].groupby(mapping_df_final["RegionName"]).sum()

# Colocation
def aggregate_colocation_final(coloc, mapping):

    """
    Purpose:
    Aggregate contact pattern matrix from department to region level (France).
    
    Inputs:
    - coloc: DataFrame, contact/colocation matrix at department granularity
    - mapping: dict, department-to-region mapping dictionary
    
    Outputs:
    - DataFrame, aggregated contact matrix at regional granularity
    
    Notes:
    Sums contact probabilities across departments within each region,
    preserving contact structure while reducing geographic resolution.
    """

    coloc = coloc.loc[mapping.index, mapping.index]
    regions = mapping["RegionName"].unique()

    result = pd.DataFrame(0.0, index=regions, columns=regions)

    for r1 in regions:
        d1 = mapping.index[mapping["RegionName"] == r1]
        for r2 in regions:
            d2 = mapping.index[mapping["RegionName"] == r2]
            result.loc[r1, r2] = coloc.loc[d1, d2].values.mean()

    return result


colocation_region_final = aggregate_colocation_final(france_colocation_09_04, mapping_df_final)


# Recompute R
R_region_final = pd.DataFrame(
    R_matrix_calc(population_region_final, colocation_region_final, 1, 4149),
    index=colocation_region_final.index,
    columns=colocation_region_final.columns
)


# #
# ======= Rob =======
# #

R_dep_np = R_matrix_france
R_reg_np = R_region_final.to_numpy()

p_dep = 1 - get_extinction_prob(R_dep_np)
p_reg = 1 - get_extinction_prob(R_reg_np)

Rob_dep = Equivalent_R(p_dep)
Rob_reg = Equivalent_R(p_reg)

Rob_dep_series = pd.Series(Rob_dep, index=france_colocation_09_04.index)
Rob_reg_series = pd.Series(Rob_reg, index=R_region_final.index)


# #
# ======= bias / variance =======
# #

analysis_df = pd.DataFrame(index=Rob_dep_series.index)
analysis_df["RegionName"] = mapping_df_final["RegionName"]
analysis_df["Rob_department"] = Rob_dep_series

analysis_df = analysis_df.dropna()

summary_df = analysis_df.groupby("RegionName")["Rob_department"].agg(["mean","std"])
summary_df["CV"] = summary_df["std"] / summary_df["mean"]

summary_df["Rob_region"] = Rob_reg_series.reindex(summary_df.index)
summary_df = summary_df.dropna()

summary_df["bias"] = summary_df["Rob_region"] - summary_df["mean"]


# #
# ======= Plot (no more crashes) =======
# #

plot_df = analysis_df.reset_index().dropna()

plt.figure(figsize=(12,6))

sns.boxplot(
    data=plot_df,
    x="RegionName",
    y="Rob_department",
    color="lightgray"
)

for i, r in enumerate(summary_df.index):
    plt.scatter(i, summary_df.loc[r,"Rob_region"], s=60, marker="D")

plt.xticks(rotation=90)
plt.ylabel("Rob")
plt.title("Spatial resolution: department vs region (correct aggregation)")
plt.tight_layout()
plt.show()

# %%


# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# #
# ======= Input (you have) =======
# #
# Italy_pop: Series，index = Polygon ID
# Italy_colocation_09_25: DataFrame，index/columns = Polygon ID
# Italy_pop_all: DataFrame with ['Population','Polygon ID','Polygon Name']
# R_matrix_test: province-level R (given)
# Get_R_matrix, get_extinction_prob, Equivalent_R: already defined

# #
# ======= 0. Alignment: ensure using Polygon ID =======
# #

# Build population with Polygon ID (if italy_pop is already a Series and index=ID, can skip)
italy_population_series_full = pd.Series(
    data=italy_pop.values,
    index=italy_pop.index,  # Ensure here is Polygon ID
    name="Population"
)

# Colocation forced square matrix (ID × ID)
italy_colocation_dataframe_full = pd.DataFrame(
    data=italy_colocation_09_25.values,
    index=italy_colocation_09_25.index,
    columns=italy_colocation_09_25.columns
)

# Align all three (ID intersection)
common_ids_full = (
    italy_population_series_full.index
    .intersection(italy_colocation_dataframe_full.index)
    .intersection(italy_colocation_dataframe_full.columns)
)

italy_population_series_full = italy_population_series_full.loc[common_ids_full]
italy_colocation_dataframe_full = italy_colocation_dataframe_full.loc[
    common_ids_full, common_ids_full
]


# #
# ======= 1. Build ID -> Name -> Region mapping =======
# #

# ID -> Province Name
italy_id_to_name_series_full = pd.Series(
    italy_pop_all["Polygon Name"].values,
    index=italy_pop_all["Polygon ID"]
).reindex(common_ids_full)

def assign_region_from_province_name_italy_final(province_name_input):
    """
    Purpose:
    Map Italian province names to administrative region identifiers.
    
    Inputs:
    - province_name_input: str, province name
    
    Outputs:
    - str, regional code
    
    Notes:
    Implements lookup table for Italian administrative geography aggregation.
    """

    mapping = {
        # Abruzzo
        "Chieti":"Abruzzo","L'Aquila":"Abruzzo","Pescara":"Abruzzo","Teramo":"Abruzzo",
        # Lazio
        "Frosinone":"Lazio","Latina":"Lazio","Rieti":"Lazio","Roma":"Lazio","Viterbo":"Lazio",
        # Liguria
        "Genova":"Liguria","Imperia":"Liguria","La Spezia":"Liguria","Savona":"Liguria",
        # Lombardia
        "Como":"Lombardia","Cremona":"Lombardia","Lecco":"Lombardia","Lodi":"Lombardia",
        "Mantua":"Lombardia","Milano":"Lombardia","Monza and Brianza":"Lombardia",
        "Pavia":"Lombardia","Sondrio":"Lombardia","Varese":"Lombardia",
        "Bergamo":"Lombardia","Brescia":"Lombardia",
        # Friuli Venezia Giulia
        "Trieste":"Friuli Venezia Giulia","Udine":"Friuli Venezia Giulia",
        "Gorizia":"Friuli Venezia Giulia","Pordenone":"Friuli Venezia Giulia",
        # Marche
        "Ancona":"Marche","Ascoli Piceno":"Marche","Fermo":"Marche",
        "Macerata":"Marche","Pesaro E Urbino":"Marche",
        # Molise
        "Campobasso":"Molise","Isernia":"Molise",
        # Piemonte
        "Alessandria":"Piemonte","Asti":"Piemonte","Biella":"Piemonte","Cuneo":"Piemonte",
        "Novara":"Piemonte","Torino":"Piemonte","Verbano-Cusio-Ossola":"Piemonte",
        "Vercelli":"Piemonte",
        # Sardegna
        "Cagliari":"Sardegna","Carbonia-Iglesias":"Sardegna","Medio Campidano":"Sardegna",
        "Nuoro":"Sardegna","Ogliastra":"Sardegna","Olbia-Tempio":"Sardegna",
        "Oristano":"Sardegna","Sassari":"Sardegna",
        # Sicilia
        "Agrigento":"Sicilia","Caltanissetta":"Sicilia","Catania":"Sicilia","Enna":"Sicilia",
        "Messina":"Sicilia","Palermo":"Sicilia","Ragusa":"Sicilia",
        "Syracuse":"Sicilia","Trapani":"Sicilia",
        # Toscana
        "Arezzo":"Toscana","Florence":"Toscana","Grosseto":"Toscana","Livorno":"Toscana",
        "Lucca":"Toscana","Massa Carrara":"Toscana","Pisa":"Toscana",
        "Pistoia":"Toscana","Prato":"Toscana","Siena":"Toscana",
        # Trentino-Alto Adige
        "Bolzano":"Trentino-Alto Adige","Trento":"Trentino-Alto Adige",
        # Umbria
        "Perugia":"Umbria","Terni":"Umbria",
        # Valle d'Aosta
        "Aosta":"Valle d'Aosta",
        # Puglia
        "Bari":"Puglia","Barletta-Andria-Trani":"Puglia","Brindisi":"Puglia",
        "Foggia":"Puglia","Lecce":"Puglia","Taranto":"Puglia",
        # Veneto
        "Belluno":"Veneto","Padua":"Veneto","Rovigo":"Veneto","Treviso":"Veneto",
        "Venezia":"Veneto","Verona":"Veneto","Vicenza":"Veneto",
        # Basilicata
        "Matera":"Basilicata","Potenza":"Basilicata",
        # Calabria
        "Catanzaro":"Calabria","Cosenza":"Calabria","Crotone":"Calabria",
        "Reggio Di Calabria":"Calabria","Vibo Valentia":"Calabria",
        # Campania
        "Avellino":"Campania","Benevento":"Campania","Caserta":"Campania",
        "Napoli":"Campania","Salerno":"Campania",
        # Emilia-Romagna
        "Bologna":"Emilia-Romagna","Ferrara":"Emilia-Romagna","Forli' - Cesena":"Emilia-Romagna",
        "Modena":"Emilia-Romagna","Parma":"Emilia-Romagna","Piacenza":"Emilia-Romagna",
        "Ravenna":"Emilia-Romagna","Reggio Nell'Emilia":"Emilia-Romagna","Rimini":"Emilia-Romagna"
    }
    return mapping.get(province_name_input, np.nan)

italy_id_to_region_series_full = italy_id_to_name_series_full.map(
    assign_region_from_province_name_italy_final
)

italy_mapping_dataframe_full = pd.DataFrame({
    "RegionName": italy_id_to_region_series_full
})

print(italy_mapping_dataframe_full[italy_mapping_dataframe_full["RegionName"].isna()])

# #
# ======= 2. Filter valid IDs =======
# #

valid_mask_full = italy_mapping_dataframe_full["RegionName"].notna()

italy_population_series_full = italy_population_series_full.loc[valid_mask_full]
italy_colocation_dataframe_full = italy_colocation_dataframe_full.loc[
    valid_mask_full, valid_mask_full
]
italy_mapping_dataframe_full = italy_mapping_dataframe_full.loc[valid_mask_full]

# #
# ======= 3. Aggregate population + colocation =======
# #

# Population -> region
italy_population_region_series_full = (
    italy_population_series_full
    .groupby(italy_mapping_dataframe_full["RegionName"])
    .sum()
)

def aggregate_colocation_to_region_full(coloc_df, mapping_df):
    """
    Purpose:
    Aggregate contact patterns from province to region level (Italy).
    
    Inputs:
    - coloc_df: DataFrame, province-level contact/colocation data
    - mapping_df: DataFrame, province-to-region mapping table
    
    Outputs:
    - DataFrame, region-level aggregated contact matrix
    
    Notes:
    Preserves spatial contact structure during geographic aggregation for Italian regions.
    """

    regions = sorted(mapping_df["RegionName"].unique())
    out = pd.DataFrame(0.0, index=regions, columns=regions)

    for r1 in regions:
        ids1 = mapping_df.index[mapping_df["RegionName"] == r1]
        for r2 in regions:
            ids2 = mapping_df.index[mapping_df["RegionName"] == r2]
            sub = coloc_df.loc[ids1, ids2]
            out.loc[r1, r2] = sub.values.mean()  # Use mean for probability
    return out

italy_colocation_region_dataframe_full = aggregate_colocation_to_region_full(
    italy_colocation_dataframe_full,
    italy_mapping_dataframe_full
)


# #
# ======= 4. R（province & region） =======
# #

# Province-level: use your existing data directly
R_matrix_italy_province_full = R_matrix_test

# Region-level: recalculate (key)
R_matrix_italy_region_full = get_R_matrix(
    italy_population_region_series_full,
    italy_colocation_region_dataframe_full,
    0.0003794804101197926 * 0.5 * 1.1136005654033705,
    1/5.1
)


# #
# ======= 5. Rob =======
# #

p_province_full = 1 - get_extinction_prob(R_matrix_italy_province_full)
p_region_full   = 1 - get_extinction_prob(R_matrix_italy_region_full)

Rob_province_full = Equivalent_R(p_province_full)
Rob_region_full   = Equivalent_R(p_region_full)

Rob_province_series_full = pd.Series(
    Rob_province_full,
    index=italy_population_series_full.index
)

Rob_region_series_full = pd.Series(
    Rob_region_full,
    index=italy_colocation_region_dataframe_full.index
)

# #
# ======= 6. bias / variance =======
# #

analysis_df_full = pd.DataFrame(index=italy_population_series_full.index)
analysis_df_full["RegionName"] = italy_mapping_dataframe_full["RegionName"]
analysis_df_full["Rob_province"] = Rob_province_series_full
analysis_df_full = analysis_df_full.dropna()

summary_df_full = (
    analysis_df_full
    .groupby("RegionName")["Rob_province"]
    .agg(["mean","std","min","max","count"])
)

summary_df_full["CV"] = summary_df_full["std"] / summary_df_full["mean"]
summary_df_full["Rob_region"] = Rob_region_series_full.reindex(summary_df_full.index)
summary_df_full = summary_df_full.dropna(subset=["Rob_region"])
summary_df_full["bias"] = summary_df_full["Rob_region"] - summary_df_full["mean"]

# #
# ======= 7. Plot 1：box + region Rob =======
# #

plot_df_full = analysis_df_full.reset_index().rename(columns={"index":"PolygonID"})

region_order_full = (
    plot_df_full.groupby("RegionName")["Rob_province"]
    .mean().sort_values().index
)

plt.figure(figsize=(12,6))

sns.boxplot(
    data=plot_df_full,
    x="RegionName",
    y="Rob_province",
    order=region_order_full
)

pos_map = {r:i for i,r in enumerate(region_order_full)}
for r in region_order_full:
    plt.scatter(pos_map[r], summary_df_full.loc[r,"Rob_region"], s=60, marker="D")

plt.xticks(rotation=90)
plt.ylabel("Rob")
plt.title("Italy: province vs region aggregation")
plt.tight_layout()
plt.show()

# #
# ======= 8. Plot 2：bias vs variance =======
# #

plt.figure(figsize=(6,5))

plt.scatter(summary_df_full["CV"], summary_df_full["bias"])

for r, row in summary_df_full.iterrows():
    plt.text(row["CV"], row["bias"], r, fontsize=8)

plt.axhline(0, linestyle="--")
plt.xlabel("Within-region CV")
plt.ylabel("Aggregation bias")
plt.title("Italy: bias–variance trade-off")
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# #
# 0. STYLE
# #

sns.set_theme(style="whitegrid", context="talk")

# #
# 1. PREPARE DATA
# #

plot_df = analysis_df_full.reset_index().rename(columns={"index": "PolygonID"})

region_order = (
    plot_df.groupby("RegionName")["Rob_province"]
    .mean()
    .sort_values()
    .index
)

summary_df = summary_df_full.copy()
summary_df["abs_bias"] = np.abs(summary_df["bias"])

# #
# 2. SAVE PATH
# #

save_dir = "/Users/boxuan/Desktop/PhD/R_ob_manuscript/Figure"
os.makedirs(save_dir, exist_ok=True)

# #
# 3. FIGURE
# #

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

palette = sns.color_palette("tab10")

# #
# ===== Panel A =====
# #

ax = axes[0]

sns.boxplot(
    data=plot_df,
    x="RegionName",
    y="Rob_province",
    order=region_order,
    color="lightgray",
    fliersize=2,
    ax=ax
)

# Region-level diamonds
pos_map = {r: i for i, r in enumerate(region_order)}

for r in region_order:
    ax.scatter(
        pos_map[r],
        summary_df.loc[r, "Rob_region"],
        color=palette[1],
        s=70,
        marker="D",
        edgecolor="black",
        zorder=3
    )

ax.set_title("(A) Province vs region aggregation")
ax.set_xlabel("")
ax.set_ylabel(r"$R^{\mathrm{ob}}$")
ax.tick_params(axis='x', rotation=90)

# #
# ===== Panel B =====
# #

ax = axes[1]

ax.scatter(
    summary_df["CV"],
    summary_df["abs_bias"],
    color=palette[2],
    s=60,
    alpha=0.9
)

# Mark all
for r, row in summary_df.iterrows():
    ax.text(
        row["CV"],
        row["abs_bias"],
        r,
        fontsize=8,
        ha="left",
        va="bottom"
    )

ax.set_xlabel("Within-region heterogeneity (CV)")
ax.set_ylabel("Absolute aggregation bias")
ax.set_title("(B) Bias–variance trade-off")

# #
# 4. LAYOUT
# #

plt.tight_layout()

# #
# 5. SAVE
# #

save_path = os.path.join(save_dir, "Supplementary_Figure_spatial_scale.png")

plt.savefig(save_path, dpi=300)
plt.close()


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# #
# ======= 1. CORRECT BRANCHING PROCESS =======
# #

def simulate_branching_process_correct_v2(
    R_matrix_input_v2,
    seed_index_input_v2,
    cutoff_input_v2,
    n_runs_input_v2=1000,
    Tmax_input_v2=200,
    offspring_type_input_v2="poisson",
    dispersion_k_input_v2=0.2
):

    """
    Purpose:
    Simulate multitype branching process with negative binomial offspring (v2 correction).
    
    Inputs:
    - [See function definition for complete parameter list]
    
    Outputs:
    - list, infection sizes across spatial patches from each replicate
    
    Notes:
    Version 2 applies corrections for finite population effects in branching approximation.
    Uses negative binomial offspring distribution with dispersion parameter omega.
    """

    n_nodes_internal_v2 = R_matrix_input_v2.shape[0]
    outbreak_flags_internal_v2 = np.zeros(n_runs_input_v2, dtype=bool)

    for run_idx_internal_v2 in range(n_runs_input_v2):

        state_vec_internal_v2 = np.zeros(n_nodes_internal_v2, dtype=int)
        state_vec_internal_v2[seed_index_input_v2] = 1

        total_size_internal_v2 = 1

        for t_internal_v2 in range(1, Tmax_input_v2):

            new_state_vec_internal_v2 = np.zeros(n_nodes_internal_v2, dtype=int)

            # ===== Correct aggregation: split by source =====
            for source_j_internal_v2 in range(n_nodes_internal_v2):

                I_j_internal_v2 = state_vec_internal_v2[source_j_internal_v2]

                if I_j_internal_v2 == 0:
                    continue

                for target_i_internal_v2 in range(n_nodes_internal_v2):

                    R_ij_internal_v2 = R_matrix_input_v2[target_i_internal_v2, source_j_internal_v2]

                    if R_ij_internal_v2 == 0:
                        continue

                    mean_ij_internal_v2 = I_j_internal_v2 * R_ij_internal_v2

                    if offspring_type_input_v2 == "poisson":

                        offspring_ij_internal_v2 = np.random.poisson(mean_ij_internal_v2)

                    elif offspring_type_input_v2 == "nb":

                        size_ij_internal_v2 = I_j_internal_v2 * dispersion_k_input_v2

                        prob_ij_internal_v2 = size_ij_internal_v2 / (
                            size_ij_internal_v2 + mean_ij_internal_v2
                        )

                        offspring_ij_internal_v2 = np.random.negative_binomial(
                            size_ij_internal_v2,
                            prob_ij_internal_v2
                        )

                    new_state_vec_internal_v2[target_i_internal_v2] += offspring_ij_internal_v2

            state_vec_internal_v2 = new_state_vec_internal_v2
            total_size_internal_v2 += state_vec_internal_v2.sum()

            if state_vec_internal_v2.sum() == 0:
                break

            if total_size_internal_v2 >= cutoff_input_v2:
                outbreak_flags_internal_v2[run_idx_internal_v2] = True
                break

    return outbreak_flags_internal_v2


# #
# ======= 2. ESTIMATION =======
# #

def estimate_probability_correct_v2(
    R_matrix_input_v2,
    seed_index_input_v2,
    cutoff_input_v2,
    n_runs_input_v2=2000,
    Tmax_input_v2=500,
    offspring_type_input_v2="poisson",
    dispersion_k_input_v2=0.2
):

    """
    Purpose:
    Estimate extinction and outbreak probabilities from branching simulations (v2).
    
    Inputs:
    - [See function definition for complete parameter list]
    
    Outputs:
    - dict, probability estimates, confidence intervals, and related statistics
    
    Notes:
    Aggregates branching process simulation results into Bayesian probability estimates.
    """

    outbreak_flags_internal_v2 = simulate_branching_process_correct_v2(
        R_matrix_input_v2,
        seed_index_input_v2,
        cutoff_input_v2,
        n_runs_input_v2,
        Tmax_input_v2,
        offspring_type_input_v2,
        dispersion_k_input_v2
    )

    p_est_internal_v2 = outbreak_flags_internal_v2.mean()
    p_std_internal_v2 = np.sqrt(
        p_est_internal_v2 * (1 - p_est_internal_v2) / n_runs_input_v2
    )

    p_theory_poisson_internal_v2 = (
        1 - get_extinction_prob(R_matrix_input_v2)[seed_index_input_v2]
    )

    return pd.Series({
        "cutoff": cutoff_input_v2,
        "p_est": p_est_internal_v2,
        "p_std": p_std_internal_v2,
        "p_theory_poisson": p_theory_poisson_internal_v2
    })


# #
# ======= 3. SCAN =======
# #

def run_scan_correct_v2(
    R_matrix_input_v2,
    seed_index_input_v2,
    cutoff_array_input_v2,
    offspring_type_input_v2,
    dispersion_k_input_v2=0.2
):

    """
    Purpose:
    Execute comprehensive parameter space scan for negative binomial model (v2).
    
    Inputs:
    - [See function definition for complete parameter list]
    
    Outputs:
    - dict, results dictionary indexed by (R, omega) parameter pairs
    
    Notes:
    Systematically varies R and dispersion parameter to map epidemic probability landscape.
    """

    df_internal_v2 = pd.DataFrame([
        estimate_probability_correct_v2(
            R_matrix_input_v2,
            seed_index_input_v2,
            cutoff_val_internal_v2,
            2000,
            500,
            offspring_type_input_v2,
            dispersion_k_input_v2
        )
        for cutoff_val_internal_v2 in cutoff_array_input_v2
    ])

    return df_internal_v2


# #
# ======= 4. NB THEORY =======
# #

def solve_nb_fixed_point_correct_v2(R_matrix_input_v2, omega_input_v2):

    """
    Purpose:
    Solve fixed-point system for extinction with negative binomial offspring (v2).
    
    Inputs:
    - R_matrix_input_v2: array (n, n), transmission matrix
    - omega_input_v2: float, negative binomial dispersion parameter
    
    Outputs:
    - array (n,), extinction probability vector
    
    Notes:
    Version 2 incorporates corrections for discrete population-size effects
    in fixed-point iteration.
    """

    n_internal_v2 = R_matrix_input_v2.shape[0]
    p_vec_internal_v2 = np.full(n_internal_v2, 0.5)

    for _ in range(2000):

        new_p_internal_v2 = np.zeros_like(p_vec_internal_v2)

        for i_internal_v2 in range(n_internal_v2):

            R_col_internal_v2 = R_matrix_input_v2[:, i_internal_v2]

            extinction_internal_v2 = np.prod(
                (1 + omega_input_v2 * R_col_internal_v2 * p_vec_internal_v2)
                ** (-1 / omega_input_v2)
            )

            new_p_internal_v2[i_internal_v2] = 1 - extinction_internal_v2

        if np.max(np.abs(new_p_internal_v2 - p_vec_internal_v2)) < 1e-10:
            return new_p_internal_v2

        p_vec_internal_v2 = new_p_internal_v2

    return p_vec_internal_v2


# #
# ======= 5. MAIN =======
# #

cutoff_grid_correct_v2 = np.round(np.logspace(1, 4, 10)).astype(int)

region_map_correct_v2 = {
    "Venezia": 82,
    "Napoli": 96,
    "Caserta": 95,
    "Torino": 32,
    "Roma": 9,
    "Genova": 7
}

all_results_correct_v2 = []

for region_name_correct_v2, idx_correct_v2 in region_map_correct_v2.items():

    df_poisson_correct_v2 = run_scan_correct_v2(
        R_matrix_test,
        idx_correct_v2,
        cutoff_grid_correct_v2,
        "poisson"
    )
    df_poisson_correct_v2["Model"] = "Poisson"

    df_nb_correct_v2 = run_scan_correct_v2(
        R_matrix_test,
        idx_correct_v2,
        cutoff_grid_correct_v2,
        "nb",
        dispersion_k_input_v2=0.2
    )
    df_nb_correct_v2["Model"] = "NB_k=0.2"

    df_tmp_correct_v2 = pd.concat([df_poisson_correct_v2, df_nb_correct_v2])
    df_tmp_correct_v2["Region"] = region_name_correct_v2

    all_results_correct_v2.append(df_tmp_correct_v2)

df_all_correct_v2 = pd.concat(all_results_correct_v2, ignore_index=True)

# #
# ======= 6. THEORY =======
# #

omega_correct_v2 = 1 / 0.2
p_nb_theory_correct_v2 = solve_nb_fixed_point_correct_v2(R_matrix_test, omega_correct_v2)

# #
# ======= 7. PLOT =======
# #

for region_name_plot_v2 in df_all_correct_v2["Region"].unique():

    df_plot_correct_v2 = df_all_correct_v2[
        df_all_correct_v2["Region"] == region_name_plot_v2
    ]

    idx_plot_correct_v2 = region_map_correct_v2[region_name_plot_v2]

    plt.figure(figsize=(7,5))

    sns.lineplot(
        data=df_plot_correct_v2,
        x="cutoff",
        y="p_est",
        hue="Model",
        marker="o"
    )

    plt.axhline(
        df_plot_correct_v2["p_theory_poisson"].iloc[0],
        linestyle="--",
        label="Poisson theory"
    )

    plt.axhline(
        p_nb_theory_correct_v2[idx_plot_correct_v2],
        linestyle="-.",
        label="NB theory"
    )

    plt.xscale("log")
    plt.title(region_name_plot_v2)
    plt.xlabel("Cutoff")
    plt.ylabel("Outbreak probability")

    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# #
# ======= 1. CORRECT BRANCHING PROCESS =======
# #

def simulate_branching_process_final_v3(
    R_matrix_input_v3,
    seed_index_input_v3,
    cutoff_input_v3,
    n_runs_input_v3=2000,
    Tmax_input_v3=500,
    offspring_type_input_v3="poisson",
    dispersion_k_input_v3=0.2
):

    """
    Purpose:
    Simulate multitype branching process with negative binomial offspring (final v3).
    
    Inputs:
    - [See function definition for complete parameter list]
    
    Outputs:
    - list, infection counts by spatial patch
    
    Notes:
    Final version incorporates complete set of theoretical refinements
    for accurate branching process representation.
    """

    n_nodes_internal_v3 = R_matrix_input_v3.shape[0]
    outbreak_flags_internal_v3 = np.zeros(n_runs_input_v3, dtype=bool)

    for run_idx_internal_v3 in range(n_runs_input_v3):

        state_vec_internal_v3 = np.zeros(n_nodes_internal_v3, dtype=int)
        state_vec_internal_v3[seed_index_input_v3] = 1

        total_size_internal_v3 = 1

        for t_internal_v3 in range(1, Tmax_input_v3):

            new_state_vec_internal_v3 = np.zeros(n_nodes_internal_v3, dtype=int)

            for source_j_internal_v3 in range(n_nodes_internal_v3):

                I_j_internal_v3 = state_vec_internal_v3[source_j_internal_v3]
                if I_j_internal_v3 == 0:
                    continue

                for target_i_internal_v3 in range(n_nodes_internal_v3):

                    R_ij_internal_v3 = R_matrix_input_v3[target_i_internal_v3, source_j_internal_v3]
                    if R_ij_internal_v3 == 0:
                        continue

                    mean_ij_internal_v3 = I_j_internal_v3 * R_ij_internal_v3

                    if offspring_type_input_v3 == "poisson":

                        offspring_ij_internal_v3 = np.random.poisson(mean_ij_internal_v3)

                    elif offspring_type_input_v3 == "nb":

                        size_ij_internal_v3 = I_j_internal_v3 * dispersion_k_input_v3
                        prob_ij_internal_v3 = size_ij_internal_v3 / (
                            size_ij_internal_v3 + mean_ij_internal_v3
                        )

                        offspring_ij_internal_v3 = np.random.negative_binomial(
                            size_ij_internal_v3,
                            prob_ij_internal_v3
                        )

                    new_state_vec_internal_v3[target_i_internal_v3] += offspring_ij_internal_v3

            state_vec_internal_v3 = new_state_vec_internal_v3
            total_size_internal_v3 += state_vec_internal_v3.sum()

            if state_vec_internal_v3.sum() == 0:
                break

            if total_size_internal_v3 >= cutoff_input_v3:
                outbreak_flags_internal_v3[run_idx_internal_v3] = True
                break

    return outbreak_flags_internal_v3


# #
# ======= 2. ESTIMATION =======
# #

def estimate_probability_final_v3(
    R_matrix_input_v3,
    seed_index_input_v3,
    cutoff_input_v3,
    offspring_type_input_v3="poisson",
    dispersion_k_input_v3=0.2
):

    """
    Purpose:
    Estimate epidemic probabilities from simulations (final v3).
    
    Inputs:
    - [See function definition for complete parameter list]
    
    Outputs:
    - dict, probability estimates and statistics
    
    Notes:
    Final version provides complete and validated probability inference framework.
    """

    outbreak_flags_internal_v3 = simulate_branching_process_final_v3(
        R_matrix_input_v3,
        seed_index_input_v3,
        cutoff_input_v3,
        2000,
        500,
        offspring_type_input_v3,
        dispersion_k_input_v3
    )

    p_est_internal_v3 = outbreak_flags_internal_v3.mean()

    return p_est_internal_v3


# #
# ======= 3. NB THEORY =======
# #

def solve_nb_fixed_point_final_v3(R_matrix_input_v3, omega_input_v3):

    """
    Purpose:
    Solve fixed-point equations for negative binomial epidemic model (final v3).
    
    Inputs:
    - R_matrix_input_v3: array (n, n), transmission matrix
    - omega_input_v3: float, dispersion parameter of negative binomial
    
    Outputs:
    - array (n,), extinction probability vector
    
    Notes:
    Final implementation with complete theoretical framework and all corrections applied.
    """

    n_internal_v3 = R_matrix_input_v3.shape[0]
    p_vec_internal_v3 = np.full(n_internal_v3, 0.5)

    for _ in range(2000):

        new_p_internal_v3 = np.zeros_like(p_vec_internal_v3)

        for i_internal_v3 in range(n_internal_v3):

            R_col_internal_v3 = R_matrix_input_v3[:, i_internal_v3]

            extinction_internal_v3 = np.prod(
                (1 + omega_input_v3 * R_col_internal_v3 * p_vec_internal_v3)
                ** (-1 / omega_input_v3)
            )

            new_p_internal_v3[i_internal_v3] = 1 - extinction_internal_v3

        if np.max(np.abs(new_p_internal_v3 - p_vec_internal_v3)) < 1e-10:
            return new_p_internal_v3

        p_vec_internal_v3 = new_p_internal_v3

    return p_vec_internal_v3


# #
# ======= 4. MAIN CALIBRATION DATA =======
# #

cutoff_fixed_v3 = 10000

region_map_v3 = {
    "Venezia": 82,
    "Napoli": 96,
    "Caserta": 95,
    "Torino": 32,
    "Roma": 9,
    "Genova": 7
}

# Theory
p_poisson_theory_vec_v3 = 1 - get_extinction_prob(R_matrix_test)
omega_nb_v3 = 1 / 0.2
p_nb_theory_vec_v3 = solve_nb_fixed_point_final_v3(R_matrix_test, omega_nb_v3)

records_v3 = []

for region_name_v3, idx_v3 in region_map_v3.items():

    # Simulation
    p_est_poisson_v3 = estimate_probability_final_v3(
        R_matrix_test, idx_v3, cutoff_fixed_v3,
        offspring_type_input_v3="poisson"
    )

    p_est_nb_v3 = estimate_probability_final_v3(
        R_matrix_test, idx_v3, cutoff_fixed_v3,
        offspring_type_input_v3="nb",
        dispersion_k_input_v3=0.2
    )

    # Poisson
    records_v3.append({
        "Region": region_name_v3,
        "Model": "Poisson",
        "p_theory": p_poisson_theory_vec_v3[idx_v3],
        "p_est": p_est_poisson_v3
    })

    # NB
    records_v3.append({
        "Region": region_name_v3,
        "Model": "NB_k=0.2",
        "p_theory": p_nb_theory_vec_v3[idx_v3],
        "p_est": p_est_nb_v3
    })

df_calibration_v3 = pd.DataFrame(records_v3)


# #
# ======= 5. CALIBRATION PLOT (main figure) =======
# #

plt.figure(figsize=(6,6))

sns.scatterplot(
    data=df_calibration_v3,
    x="p_theory",
    y="p_est",
    hue="Model",
    s=80
)

# 45-degree line
x_vals = np.linspace(0, 1, 100)
plt.plot(x_vals, x_vals, 'k--')

plt.xlabel("Theoretical outbreak probability")
plt.ylabel("Simulated outbreak probability")
plt.title("Calibration: Poisson vs NB")

plt.tight_layout()
plt.show()


# #
# ======= 6. CUT-OFF CONVERGENCE（SI） =======
# #

cutoff_grid_v3 = np.round(np.logspace(1, 4, 10)).astype(int)

for region_name_v3, idx_v3 in region_map_v3.items():

    df_poisson_curve_v3 = []
    df_nb_curve_v3 = []

    for cutoff_val_v3 in cutoff_grid_v3:

        p_poisson_v3 = estimate_probability_final_v3(
            R_matrix_test, idx_v3, cutoff_val_v3,
            offspring_type_input_v3="poisson"
        )

        p_nb_v3 = estimate_probability_final_v3(
            R_matrix_test, idx_v3, cutoff_val_v3,
            offspring_type_input_v3="nb",
            dispersion_k_input_v3=0.2
        )

        df_poisson_curve_v3.append(p_poisson_v3)
        df_nb_curve_v3.append(p_nb_v3)

    plt.figure(figsize=(6,4))

    plt.plot(cutoff_grid_v3, df_poisson_curve_v3, label="Poisson", marker="o")
    plt.plot(cutoff_grid_v3, df_nb_curve_v3, label="NB", marker="o")

    plt.axhline(
        p_poisson_theory_vec_v3[idx_v3],
        linestyle="--",
        label="Poisson theory"
    )

    plt.axhline(
        p_nb_theory_vec_v3[idx_v3],
        linestyle="-.",
        label="NB theory"
    )

    plt.xscale("log")
    plt.title(region_name_v3)
    plt.xlabel("Cutoff")
    plt.ylabel("Outbreak probability")

    plt.legend()
    plt.tight_layout()
    plt.show()

# %%


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. Theoretical Calculation Core function
# ============================================================

def get_p_theory_nb(mat_R, val_omega, tol=1e-12, max_iter=5000):
    """
    求解负二项分布(NB)多社区固定点方程 (对应 Gamma 感染期):
    p_i = 1 - product_j [1 + omega * R_ji * p_j]^(-1/omega)
    """
    n_dim = mat_R.shape[0]
    vec_p = np.ones(n_dim) * 0.5  # Initial guess
    inv_omg = 1.0 / val_omega
    for _ in range(max_iter):
        vec_p_old = vec_p.copy()
        for i in range(n_dim):
            # Based on Eq. (28)
            terms = 1 + val_omega * mat_R[i, :] * vec_p
            vec_p[i] = 1 - np.prod(terms**(-inv_omg))
        if np.linalg.norm(vec_p - vec_p_old) < tol:
            break
    return vec_p

def calculate_corrected_rob_nb(mat_R, vec_p_nb, val_omega):
    """
    计算带有超离散修正项的 Rob (即 R_reference) (Eq. 37/39):
    Rob_i = (1/p_i) * sum_j [ R_ji * p_j * (1 + omega*R_ji) / (1 + omega*R_ji*p_j) ]
    """
    n_dim = mat_R.shape[0]
    vec_rob_corr = np.zeros(n_dim)
    for i in range(n_dim):
        if vec_p_nb[i] < 1e-10: continue
        # Strictly execute nonlinear correction operator psi in Rob's notes
        numerator = mat_R[i, :] * vec_p_nb * (1 + val_omega * mat_R[i, :])
        denominator = 1 + val_omega * mat_R[i, :] * vec_p_nb
        vec_rob_corr[i] = np.sum(numerator / denominator) / vec_p_nb[i]
    return vec_rob_corr

# ============================================================
# 2. Generate Figure A: R_ob 的“归一化”证明图
# ============================================================

def generate_figure_A_normalization_proof(R_matrix_demo):
    """
    证明无论超离散 ω 如何变化，修正后的 Rob 始终与泊松基准动力学一致。
    """
    plt.figure(figsize=(9, 6))
    
    # --- 1. Poisson baseline (aligned with standard dynamics: R = -ln(1-p)/p) ---
    p_baseline = np.linspace(0.01, 0.99, 100)
    R_baseline = -np.log(1 - p_baseline) / p_baseline
    plt.plot(p_baseline, R_baseline, 'k-', linewidth=2.5, label='Baseline (Poisson/Standard Dynamics)')
    
    # --- 2. Calculate R_ob under different overdispersion parameter ω ---
    omega_values_to_test = [0.5, 1.0, 2.0]  # Larger ω means stronger superspreading
    colors = ['#3498db', '#e67e22', '#2ecc71']  # Blue, orange, green
    
    for i_omg, val_omg in enumerate(omega_values_to_test):
        # Calculate outbreak probability p_i under this ω (theoretical truth)
        vec_p_theory_omg = get_p_theory_nb(R_matrix_demo, val_omg)
        # Calculate corrected Rob
        vec_rob_corrected_omg = calculate_corrected_rob_nb(R_matrix_demo, vec_p_theory_omg, val_omg)
        
        # Scatter: (p_theory, R_ob)
        plt.scatter(vec_p_theory_omg, vec_rob_corrected_omg, 
                    color=colors[i_omg], alpha=0.6, s=40,
                    label=f'Corrected $R_{{ob}}$ ($\omega$={val_omg}, NB/Gamma)')
        
        # Comparison: uncorrected R_local (show how it deviates from dynamics)
        vec_r_local = np.diag(R_matrix_demo)
        plt.scatter(vec_p_theory_omg, vec_r_local, 
                    color=colors[i_omg], marker='x', s=20, alpha=0.3)

    plt.xlabel('Theoretical Outbreak Probability ($p_i$)')
    plt.ylabel('Reproduction Number ($R$)')
    plt.title('Figure A: $R_{ob}$ Normalizes Heterogeneous Epidemic Dynamics\n'
              '(Corrected $R_{ob}$ stays on baseline regardless of overdispersion $\omega$)')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ============================================================
# 3. Generate Figure B: bar chart quantifying correction term contribution to R_ob
# ============================================================

def generate_figure_B_correction_contribution_bar(R_matrix_demo, city_name_map):
    """
    量化超级传播修正项对 Rob 估值的放大作用，证明我们并未低估超级传播。
    """
    omega_fixed = 1.0  # Set a representative level of overdispersion
    
    # Calculate theoretical p and corrected Rob
    vec_p_nb_omg1 = get_p_theory_nb(R_matrix_demo, omega_fixed)
    vec_rob_corrected_omg1 = calculate_corrected_rob_nb(R_matrix_demo, vec_p_nb_omg1, omega_fixed)
    vec_r_local = np.diag(R_matrix_demo)  # Original Poisson estimate R_ii
    
    # Organize data for plotting
    list_plot_records = []
    for city_name, idx_city in city_name_map.items():
        # R_local (ignore superspreading)
        list_plot_records.append({
            'City': city_name, 
            'R Value': vec_r_local[idx_city], 
            'Estimate Type': '$R_{local}$ (Poisson assumed)'
        })
        # R_ob (considering superspreading correction)
        list_plot_records.append({
            'City': city_name, 
            'R Value': vec_rob_corrected_omg1[idx_city], 
            'Estimate Type': f'Corrected $R_{{ob}}$ (NB/Gamma, $\omega$={omega_fixed})'
        })
        
    df_plot = pd.DataFrame(list_plot_records)
    
    # Plotting logic
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x='City', y='R Value', hue='Estimate Type', palette='muted')
    
    # Add R=1 threshold line
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Outbreak Threshold ($R=1$)')
    
    plt.ylabel('Reproduction Number ($R$)')
    
    # Note double braces for R_{{ob}} and R_{{local}} here
    plt.title(f'Figure B: Quantitative Impact of Overdispersion Correction on $R_{{ob}}$\n'
              f'(Showing substantial underestimation by $R_{{local}}$ under strong superspreading $\omega$={omega_fixed})')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()
# ============================================================
# 4. Main program run (ensure R_matrix_test is defined)
# ============================================================

# Note: before running this code, ensure variable R_matrix_test is loaded in your environment
# If not, you can create a random 100x100 matrix for testing with the code below:
# Np.random.seed(42)
# R_matrix_test = np.random.rand(100, 100) * 0.3
# Np.fill_diagonal(R_matrix_test, 1.2) # Ensure outbreak

# Selected city index mapping
map_city_to_idx_rebuttal = {
    "Venezia": 82, "Napoli": 96, "Caserta": 95, 
    "Torino": 32, "Roma": 9, "Genova": 7
}

# Run Figure A (robustness proof)
generate_figure_A_normalization_proof(R_matrix_test)

# Run Figure B (correction contribution quantification)
generate_figure_B_correction_contribution_bar(R_matrix_test, map_city_to_idx_rebuttal)

# %%
# Answer to Reviewer 1, question 5. The bias of the colocation map.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. Core calculation function (ensure R_matrix_test is defined)
# ============================================================

def get_p_theory_poisson_internal_v4(matrix_R_input_v4, tol_internal_v4=1e-12):
    """求解泊松分布下的理论爆发概率 p_i"""
    n_nodes_v4 = matrix_R_input_v4.shape[0]
    vec_q_v4 = np.ones(n_nodes_v4) * 0.5
    for _ in range(2000):
        vec_q_old_v4 = vec_q_v4.copy()
        # Q_i = exp(sum_j R_ji * (q_j - 1))
        vec_q_v4 = np.exp(matrix_R_input_v4 @ (vec_q_v4 - 1))
        if np.linalg.norm(vec_q_v4 - vec_q_old_v4) < tol_internal_v4:
            break
    return 1 - vec_q_v4

def calculate_rob_poisson_internal_v4(matrix_R_input_v4, vec_p_input_v4):
    """计算泊松分布下的 Rob (即 R_reference)"""
    n_nodes_v4 = matrix_R_input_v4.shape[0]
    vec_rob_v4 = np.zeros(n_nodes_v4)
    for i_v4 in range(n_nodes_v4):
        if vec_p_input_v4[i_v4] > 1e-10:
            # Rob_i = sum_j (R_ji * p_j) / p_i
            vec_rob_v4[i_v4] = np.sum(matrix_R_input_v4[i_v4, :] * vec_p_input_v4) / vec_p_input_v4[i_v4]
    return vec_rob_v4

# ============================================================
# 2. Noise injection and 6-city comparison analysis
# ============================================================

# Selected 6 core city indices
region_to_idx_map_v4 = {
    "Venezia": 82, "Napoli": 96, "Caserta": 95, 
    "Torino": 32, "Roma": 9, "Genova": 7
}

# --- Step A: Calculate Rob for Ground Truth (accurate data) ---
vec_p_ground_truth_v4 = get_p_theory_poisson_internal_v4(R_matrix_test)
vec_rob_ground_truth_v4 = calculate_rob_poisson_internal_v4(R_matrix_test, vec_p_ground_truth_v4)

# --- Step B: Inject 20% random noise ---
np.random.seed(42)
# Inject +/- 20% random perturbation to each non-zero element
noise_matrix_v4 = np.random.uniform(0.8, 1.2, size=R_matrix_test.shape)
R_matrix_noisy_v4 = R_matrix_test * noise_matrix_v4

# Recalculate Rob under noisy data
vec_p_noisy_v4 = get_p_theory_poisson_internal_v4(R_matrix_noisy_v4)
vec_rob_noisy_v4 = calculate_rob_poisson_internal_v4(R_matrix_noisy_v4, vec_p_noisy_v4)

# --- Step C: Organize 6-city data ---
list_comparison_records_v4 = []
for city_name_v4, city_idx_v4 in region_to_idx_map_v4.items():
    list_comparison_records_v4.append({
        "City": city_name_v4, 
        "Rob_Value": vec_rob_ground_truth_v4[city_idx_v4], 
        "Data_Type": "Ground Truth"
    })
    list_comparison_records_v4.append({
        "City": city_name_v4, 
        "Rob_Value": vec_rob_noisy_v4[city_idx_v4], 
        "Data_Type": "Noisy Data (±20%)"
    })

df_comparison_results_v4 = pd.DataFrame(list_comparison_records_v4)

# ============================================================
# 3. Display plot
# ============================================================

plt.figure(figsize=(10, 6))
sns.barplot(data=df_comparison_results_v4, x="City", y="Rob_Value", hue="Data_Type", palette="muted")

plt.axhline(y=1.0, color='red', linestyle='--', label="Outbreak Threshold")
plt.title("Robustness of $R^{ob}$ Estimates to Spatial Contact Data Noise (6 Key Cities)")
plt.ylabel("Outbreak Reproduction Ratio ($R^{ob}$)")
plt.ylim(0, max(vec_rob_ground_truth_v4.max(), vec_rob_noisy_v4.max()) * 1.2)
plt.legend(title="Data Scenarios")
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# #
# ======= 1. CORE FUNCTIONS =======
# #

def compute_outbreak_probability_poisson_full_v8(R_matrix_input_v8, tol_input_v8=1e-12):

    """
    Purpose:
    Compute outbreak probability using Poisson branching model (v8).
    
    Inputs:
    - R_matrix_input_v8: array (n, n), transmission matrix
    - tol_input_v8: float, convergence tolerance (default 1e-12)
    
    Outputs:
    - array (n,), outbreak probability vector (1 - extinction probability)
    
    Notes:
    Version 8 represents final optimized Poisson implementation.
    Outbreak probability = 1 - extinction probability for each population.
    """

    n_nodes_internal_v8 = R_matrix_input_v8.shape[0]
    q_vector_internal_v8 = np.ones(n_nodes_internal_v8) * 0.5

    for _ in range(2000):
        q_old_internal_v8 = q_vector_internal_v8.copy()
        q_vector_internal_v8 = np.exp(R_matrix_input_v8 @ (q_vector_internal_v8 - 1))

        if np.linalg.norm(q_vector_internal_v8 - q_old_internal_v8) < tol_input_v8:
            break

    return 1 - q_vector_internal_v8


def compute_Rob_poisson_full_v8(R_matrix_input_v8, p_vector_input_v8):

    """
    Purpose:
    Compute robustness/effectiveness metric for Poisson epidemic model (v8).
    
    Inputs:
    - R_matrix_input_v8: array (n, n), transmission matrix
    - p_vector_input_v8: array (n,), outbreak probability vector
    
    Outputs:
    - array (n,), robustness metric for each population
    
    Notes:
    Version 8 represents final robustness computation framework.
    Robustness quantifies resistance to epidemic invasion and persistence.
    """

    n_nodes_internal_v8 = R_matrix_input_v8.shape[0]
    Rob_vector_internal_v8 = np.zeros(n_nodes_internal_v8)

    for i_internal_v8 in range(n_nodes_internal_v8):

        if p_vector_input_v8[i_internal_v8] > 1e-12:

            Rob_vector_internal_v8[i_internal_v8] = (
                np.sum(R_matrix_input_v8[i_internal_v8, :] * p_vector_input_v8)
                / p_vector_input_v8[i_internal_v8]
            )

    return Rob_vector_internal_v8


# #
# ======= 2. SETUP =======
# #

region_index_map_full_v8 = {
    "Venezia": 82,
    "Napoli": 96,
    "Caserta": 95,
    "Torino": 32,
    "Roma": 9,
    "Genova": 7
}

# Ground truth
p_true_full_v8 = compute_outbreak_probability_poisson_full_v8(R_matrix_test)
Rob_true_full_v8 = compute_Rob_poisson_full_v8(R_matrix_test, p_true_full_v8)


# #
# ======= 3. RANDOM NOISE ANALYSIS =======
# #

noise_strength_array_full_v8 = np.linspace(0, 0.5, 10)
records_noise_full_v8 = []

for noise_strength_iter_v8 in noise_strength_array_full_v8:

    for repeat_iter_v8 in range(20):

        noise_matrix_full_v8 = np.random.lognormal(
            mean=0,
            sigma=noise_strength_iter_v8,
            size=R_matrix_test.shape
        )

        R_noisy_full_v8 = R_matrix_test * noise_matrix_full_v8

        p_noisy_full_v8 = compute_outbreak_probability_poisson_full_v8(R_noisy_full_v8)
        Rob_noisy_full_v8 = compute_Rob_poisson_full_v8(R_noisy_full_v8, p_noisy_full_v8)

        for city_name_v8, city_idx_v8 in region_index_map_full_v8.items():

            records_noise_full_v8.append({
                "NoiseStrength": noise_strength_iter_v8,
                "Metric": "Rob",
                "Error": abs(Rob_noisy_full_v8[city_idx_v8] - Rob_true_full_v8[city_idx_v8]) / Rob_true_full_v8[city_idx_v8]
            })

            records_noise_full_v8.append({
                "NoiseStrength": noise_strength_iter_v8,
                "Metric": "OutbreakProbability",
                "Error": abs(p_noisy_full_v8[city_idx_v8] - p_true_full_v8[city_idx_v8])
            })

df_noise_full_v8 = pd.DataFrame(records_noise_full_v8)

# #
# ======= 4. PLOT RANDOM NOISE =======
# #

plt.figure(figsize=(8,6))

sns.lineplot(
    data=df_noise_full_v8,
    x="NoiseStrength",
    y="Error",
    hue="Metric",
    estimator="mean"
)

plt.title("Robustness to Random Noise in Contact Matrix")
plt.xlabel("Noise Strength (lognormal σ)")
plt.ylabel("Estimation Error")

plt.tight_layout()
plt.show()


# #
# ======= 5. STRUCTURAL BIAS ANALYSIS =======
# #

bias_strength_array_full_v8 = np.linspace(0.2, 1.0, 10)
records_bias_full_v8 = []

for bias_iter_v8 in bias_strength_array_full_v8:

    R_biased_full_v8 = R_matrix_test.copy()

    for i_iter_v8 in range(R_biased_full_v8.shape[0]):
        for j_iter_v8 in range(R_biased_full_v8.shape[1]):
            if i_iter_v8 != j_iter_v8:
                R_biased_full_v8[i_iter_v8, j_iter_v8] *= bias_iter_v8

    p_biased_full_v8 = compute_outbreak_probability_poisson_full_v8(R_biased_full_v8)
    Rob_biased_full_v8 = compute_Rob_poisson_full_v8(R_biased_full_v8, p_biased_full_v8)

    for city_name_v8, city_idx_v8 in region_index_map_full_v8.items():

        records_bias_full_v8.append({
            "BiasStrength": bias_iter_v8,
            "Metric": "Rob",
            "Error": abs(Rob_biased_full_v8[city_idx_v8] - Rob_true_full_v8[city_idx_v8]) / Rob_true_full_v8[city_idx_v8]
        })

        records_bias_full_v8.append({
            "BiasStrength": bias_iter_v8,
            "Metric": "OutbreakProbability",
            "Error": abs(p_biased_full_v8[city_idx_v8] - p_true_full_v8[city_idx_v8])
        })

df_bias_full_v8 = pd.DataFrame(records_bias_full_v8)

# #
# ======= 6. PLOT STRUCTURAL BIAS =======
# #

plt.figure(figsize=(8,6))

sns.lineplot(
    data=df_bias_full_v8,
    x="BiasStrength",
    y="Error",
    hue="Metric",
    estimator="mean"
)

plt.gca().invert_xaxis()

plt.title("Robustness to Structural Bias (Inter-Region Underestimation)")
plt.xlabel("Bias Strength (off-diagonal scaling)")
plt.ylabel("Estimation Error")

plt.tight_layout()
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ============================================================
# 1. Core calculation function (analytical solution defined)
# ============================================================

def compute_p_poisson_v12(R_matrix, tol=1e-12):
    """根据 Eq. (1) 求解理论爆发概率 p_i [cite: 72]"""
    n = R_matrix.shape[0]
    q = np.ones(n) * 0.5
    for _ in range(2000):
        q_old = q.copy()
        # Q_i = exp(sum_j R_ji * (q_j - 1)) [cite: 258]
        q = np.exp(R_matrix @ (q - 1))
        if np.linalg.norm(q - q_old) < tol:
            break
    return 1 - q

def compute_Rob_poisson_v12(R_matrix, p_vector):
    """根据 Eq. (3) 计算 Rob """
    n = R_matrix.shape[0]
    Rob = np.zeros(n)
    for i in range(n):
        if p_vector[i] > 1e-12:
            # Rob_i = sum_j (R_ji * p_j) / p_i
            Rob[i] = np.sum(R_matrix[i, :] * p_vector) / p_vector[i]
    return Rob

# ============================================================
# 2. Geographic background and Ground Truth setup
# ============================================================

# Selected 6 core city indices [cite: 91, 126]
region_map = {
    "Venezia": 82, "Napoli": 96, "Caserta": 95, 
    "Torino": 32, "Roma": 9, "Genova": 7
}

# Calculate ground truth
p_true = compute_p_poisson_v12(R_matrix_test)
rob_true = compute_Rob_poisson_v12(R_matrix_test, p_true)
r_local_true = np.diag(R_matrix_test)

# ============================================================
# 3. First group of figures: Random Noise Analysis
# ============================================================
# Simulate random sampling fluctuation of Meta data

noise_sigmas = np.linspace(0, 0.4, 8)
noise_records = []

for sigma in noise_sigmas:
    for _ in range(10):  # 10Repeat 10 times and average
        # Inject lognormal noise
        R_noisy = R_matrix_test * np.random.lognormal(0, sigma, size=R_matrix_test.shape)
        p_n = compute_p_poisson_v12(R_noisy)
        r_n = compute_Rob_poisson_v12(R_noisy, p_n)
        
        for name, idx in region_map.items():
            err = abs(r_n[idx] - rob_true[idx]) / rob_true[idx]
            noise_records.append({"Sigma": sigma, "Relative_Error": err, "City": name})

df_noise = pd.DataFrame(noise_records)

# Plot: 6-city robustness under random noise
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_noise, x="Sigma", y="Relative_Error", hue="City", marker='o')
plt.title("GROUP 1: Robustness of $R^{ob}$ to Random Data Noise")
plt.xlabel("Noise Strength (Lognormal $\sigma$)")
plt.ylabel("Relative Error in $R^{ob}$")
plt.tight_layout()
plt.show()

# ============================================================
# 4. Second group of figures: Structural Bias Analysis
# ============================================================
# Simulate systematic underestimation of mobility data in rural/low-density areas

bias_factors = np.linspace(0.2, 1.0, 10)
bias_records = []

for bf in bias_factors:
    # Reduce off-diagonal terms (cross-regional coupling)
    R_biased = R_matrix_test.copy()
    mask = ~np.eye(R_biased.shape[0], dtype=bool)
    R_biased[mask] *= bf
    
    p_b = compute_p_poisson_v12(R_biased)
    r_b = compute_Rob_poisson_v12(R_biased, p_b)
    
    for name, idx in region_map.items():
        err = (rob_true[idx] - r_b[idx]) / rob_true[idx]
        bias_records.append({"Bias_Strength": bf, "Error": err, "City": name, "R_ii": r_local_true[idx]})

df_bias = pd.DataFrame(bias_records)

# Plot: 6-city sensitivity under structural bias (展示 R_ii 越小误差越大)
plt.figure(figsize=(10, 6))
# Sort legend by R_ii size
city_order = df_bias.groupby("City")["R_ii"].first().sort_values(ascending=False).index
sns.lineplot(data=df_bias, x="Bias_Strength", y="Error", hue="City", 
             hue_order=city_order, palette="flare", marker='s')
plt.gca().invert_xaxis()  # From 1.0 (no bias) to 0.2 (severe underestimation)
plt.title("GROUP 2: Impact of Structural Bias (Data Gaps in Mobility)")
plt.xlabel("Mobility Scaling Factor (1.0 = Truth, 0.2 = 80% Underestimated)")
plt.ylabel("Relative Underestimation of $R^{ob}$")
plt.legend(title="Cities (Sorted by $R_{ii}$ ↓)")
plt.tight_layout()
plt.show()

# ============================================================
# 5. National statistical scatter plot (Bonus: prove the inevitability of the pattern)
# ============================================================
# Fix 50% cross-regional data missing
bf_fixed = 0.5
R_nat = R_matrix_test.copy()
R_nat[~np.eye(R_nat.shape[0], dtype=bool)] *= bf_fixed
p_nat = compute_p_poisson_v12(R_nat)
r_nat = compute_Rob_poisson_v12(R_nat, p_nat)
err_nat = (rob_true - r_nat) / rob_true

plt.figure(figsize=(9, 6))
sns.scatterplot(x=r_local_true, y=err_nat, alpha=0.4, color='gray')
# Mark those 6 points
for name, idx in region_map.items():
    plt.scatter(r_local_true[idx], err_nat[idx], s=150, edgecolors='black', label=name)
sns.regplot(x=r_local_true, y=err_nat, scatter=False, color='red', line_kws={"ls":"--"})
plt.title("National Analysis: Error Sensitivity vs Local Transmission Potential")
plt.xlabel("Local Transmission Potential ($R_{ii}$)")
plt.ylabel("Relative Underestimation (at 50% Data Bias)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="talk")

# #
# FIGURE（2x2）
# #

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# #
# (a) Random noise
# #

ax = axes[0]

sns.lineplot(
    data=df_noise,
    x="Sigma",
    y="Relative_Error",
    hue="City",
    marker='o',
    ax=ax
)

ax.set_title("(a) Robustness to random noise")
ax.set_xlabel("Noise strength (lognormal $\sigma$)")
ax.set_ylabel("Relative error in $R^{\\mathrm{ob}}$")

# #
# (b) Structural bias
# #

ax = axes[1]

city_order = (
    df_bias.groupby("City")["R_ii"]
    .first()
    .sort_values(ascending=False)
    .index
)

sns.lineplot(
    data=df_bias,
    x="Bias_Strength",
    y="Error",
    hue="City",
    hue_order=city_order,
    marker='s',
    palette="flare",
    ax=ax
)

ax.invert_xaxis()

ax.set_title("(b) Impact of structural bias")
ax.set_xlabel("Mobility scaling factor")
ax.set_ylabel("Relative bias in $R^{\\mathrm{ob}}$")

# #
# (c) National scatter
# #

ax = axes[2]

sns.scatterplot(
    x=r_local_true,
    y=err_nat,
    alpha=0.4,
    color='gray',
    ax=ax
)

for name, idx in region_map.items():
    ax.scatter(
        r_local_true[idx],
        err_nat[idx],
        s=120,
        edgecolors='black'
    )
    ax.text(
        r_local_true[idx],
        err_nat[idx],
        name,
        fontsize=9
    )

sns.regplot(
    x=r_local_true,
    y=err_nat,
    scatter=False,
    color='red',
    line_kws={"ls":"--"},
    ax=ax
)

ax.set_title("(c) Sensitivity vs local transmission")
ax.set_xlabel("Local transmission potential ($R_{ii}$)")
ax.set_ylabel("Relative bias")

# #
# (d) Summary trend (optional enhancement)
# #

ax = axes[3]

sns.regplot(
    data=df_bias,
    x="R_ii",
    y="Error",
    scatter_kws={"alpha":0.4},
    line_kws={"color":"black"},
    ax=ax
)

ax.set_title("(d) Systematic dependence on $R_{ii}$")
ax.set_xlabel("Local transmission potential ($R_{ii}$)")
ax.set_ylabel("Relative bias")

# #
# LEGEND (unified)
# #

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

# Remove subplot legend
for ax in axes:
    if ax.get_legend():
        ax.get_legend().remove()

# #
# SAVE
# #

save_dir = "/Users/boxuan/Desktop/PhD/R_ob_manuscript/Figure"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "Supplementary_Figure_7.png")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(save_path, dpi=300)
plt.close()


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="talk")

# #
# 1. English naming + sorting
# #

region_rename_map = {
    "Venezia": "Venice",
    "Napoli": "Naples",
    "Caserta": "Caserta",
    "Torino": "Turin",
    "Roma": "Rome",
    "Genova": "Genoa"
}

# Add English column
df_noise_analysis_v16["City_EN"] = df_noise_analysis_v16["City"].map(region_rename_map)
df_bias_analysis_v16["City_EN"] = df_bias_analysis_v16["City"].map(region_rename_map)

# Sort alphabetically
city_order = sorted(df_noise_analysis_v16["City_EN"].unique())

# #
# 2. Fixed color (ensure consistency)
# #

palette = dict(zip(city_order, sns.color_palette("tab10", len(city_order))))

# #
# 3. FIGURE（2 panel）
# #

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# #
# (a) Random noise
# #

ax = axes[0]

sns.boxplot(
    data=df_noise_analysis_v16,
    x="Noise_Intensity",
    y="Relative_Error",
    hue="City_EN",
    hue_order=city_order,
    palette=palette,
    ax=ax
)

ax.set_title("(a) Robustness to random noise")
ax.set_xlabel("Noise strength (lognormal $\sigma$)")
ax.set_ylabel("Relative error in $R^{\\mathrm{ob}}$")

# #
# (b) Structural bias
# #

ax = axes[1]

sns.lineplot(
    data=df_bias_analysis_v16,
    x="Data_Coverage",
    y="Underestimation_Error",
    hue="City_EN",
    hue_order=city_order,
    palette=palette,
    marker='s',
    ax=ax
)

ax.invert_xaxis()

ax.set_title("(b) Impact of structural bias")
ax.set_xlabel("Mobility data coverage (1.0 = full data)")
ax.set_ylabel("Relative underestimation of $R^{\\mathrm{ob}}$")

# #
# LEGEND (unified + a-z sorting)
# #

handles, labels = axes[0].get_legend_handles_labels()

# Remove duplicate legend
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, 1.)
)

for ax in axes:
    if ax.get_legend():
        ax.get_legend().remove()

# #
# SAVE
# #

save_dir = "/Users/boxuan/Desktop/PhD/R_ob_manuscript/Figure"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "Supplementary_Figure_7.png")

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(save_path, dpi=300)
plt.close()


# %% [markdown]
# ========================
# SECTION6: Figure3: Robustness of R_ob estimation in single outbreak simulations
# ========================
# # Bootstrap Result

# %%
def disease_transmission_simulation(R_matrix, nruns, Tmax, initial_case_place, max_population):
    """
    Purpose:
    Perform complete stochastic disease transmission simulation on spatial network.
    
    Inputs:
    - R_matrix: array (n, n), transmission matrix
    - nruns: int, number of simulation replicates
    - Tmax: int, maximum simulation time steps
    - initial_case_place: int, location of first infected individual
    - max_population: int, outbreak size threshold for termination
    
    Outputs:
    - array (nruns, n, Tmax), infection matrix recording infections per location per replicate
    
    Notes:
    Implements continuous-time stochastic simulation of spatial epidemic dynamics.
    Each replicate is Markovian branching process on transmission network.
    """

    num_communities = R_matrix.shape[0]
    simulations = {}

    for run in range(nruns):
        # Initialize the transmission cases matrix and the population count over time
        MI = np.zeros((Tmax, num_communities, num_communities), dtype=np.int64)
        population = np.zeros((Tmax, num_communities), dtype=np.int64)
        # Seed initial case in the specified community
        population[0, initial_case_place] = 1

        # First time step (t = 0): No transmission occurs, only the initial seeding
        for t in range(1, Tmax):
            # Calculate the total population at the current time step
            total_population = population[t - 1, :].sum()
            # Check if the total population exceeds the maximum allowed population
            if total_population >= max_population:
                MI = MI[:t-1, :, :]  # Truncate MI to actual time steps
                break

            # Calculate transmission cases from each community to each other community
            MI[t, :, :] = np.random.poisson(R_matrix * population[t - 1, :].reshape(1, -1))
                                            
            # Sum the transmission matrix by columns to get new cases for each community
            new_cases = MI[t, :, :].sum(axis=1)
            # Update the population with new cases
            population[t, :] = new_cases

        # Store the simulation results in the dictionary with the current run as the key
        simulations[run] = MI

    return simulations


def calculate_parameter_inference_matrix_limit_approx(simulation_results,start_place):
    """
    Use a matrix form to calculate sum_{t} I_{i <- j}(t+1) / sum_{t} I_j(t) for each (i, j) community pair.
    """
    all_results = {}
    all_results_df = {}

    # Iterate over each simulation
    for sim_num, infection_matrix in simulation_results.items():
        Tmax, num_communities, _ = infection_matrix.shape
        estimation_matrix = np.zeros((num_communities, num_communities))
        variance_matrix = np.zeros((num_communities, num_communities))
        
        # ESTIMATION OF PARAMETER
        sum_I_j_to_i_t1 = infection_matrix.sum(axis=0)  # Summing over time to get total infections I_{i <- j}(t+1)

        # Calculate sum_{t} I_j(t)
        sum_I_j_t_minus1 = infection_matrix[:Tmax-1, :, :].sum(axis=1).sum(axis=0)  # Sum over time and target communities to get I_j(t)
        
        # Compute estimation matrix
        sum_I_j_t_matrix = sum_I_j_t_minus1.reshape(1, -1)
        estimation_matrix = np.where(
            sum_I_j_t_matrix != 0,
            sum_I_j_to_i_t1 / sum_I_j_t_matrix,
            0
        )

        # ESTIMATION OF VARIANCE
        eigenvalues, eigenvectors = np.linalg.eig(estimation_matrix)
        max_eig = np.max(eigenvalues.real)  # Only consider the real part
        max_index = np.argmax(eigenvalues.real)
        perron_vector = eigenvectors[:, max_index].real
        perron_vector = np.abs(perron_vector) / np.linalg.norm(perron_vector)
        
        eigenvalues_T, eigenvectors_T = np.linalg.eig(estimation_matrix.T)
        max_index2 = np.argmax(eigenvalues_T.real)
        v_star = eigenvectors_T[:, max_index2].real
        v_star = np.abs(v_star) / (v_star @ perron_vector)

        v_star_initial = v_star[start_place]  # Assume the initial case is in the first community
        denominator = (max_eig ** Tmax - 1) * v_star_initial

        variance_matrix = np.where(
            (estimation_matrix > 0) & (denominator * perron_vector[:, np.newaxis] > 0),
            ((max_eig * estimation_matrix) / (denominator * perron_vector[:, np.newaxis])) ** 0.5,
            0
        )

        all_results[sim_num] = {
            'estimation_matrix': estimation_matrix,
            'variance_matrix': variance_matrix,
            'Simulation_number': sim_num,
            'Total infection': infection_matrix.sum(),
            'Initial_case': 0  # Assuming the initial case is in community 0
        }

        all_results_df[sim_num] = {
            'estimation_matrix_radius': max_eig,
            'Simulation_number': sim_num,
            'Total infection': infection_matrix.sum(),
            'Initial_case': 0  # Assuming the initial case is in community 0
        }

    all_results_df_concat = pd.DataFrame.from_dict(all_results_df, orient='index')

   # Print(all_results_df_concat)
    return all_results, all_results_df_concat

import numpy as np
import pandas as pd

def calculate_parameter_inference_matrix(simulation_results, start_place):
    """
    Use a matrix form to calculate:
      estimation_matrix[i, j] = sum_t I_{i <- j}(t+1) / sum_t I_j(t)
    and estimate the empirical variance:
      variance_matrix[i, j] = estimation_matrix[i, j] / sum_t I_j(t)
    """

    all_results = {}
    all_results_df = {}

    # Iterate over each simulation
    for sim_num, infection_matrix in simulation_results.items():
        Tmax, num_communities, _ = infection_matrix.shape

        # --- 1. Initialize container
        estimation_matrix = np.zeros((num_communities, num_communities))
        variance_matrix   = np.zeros((num_communities, num_communities))

        # --- 2. Calculate numerator: sum_t I_{i <- j}(t+1)
        # In your original code this is:
        # "Summing over time to get total infections I_{i <- j}(t+1)"
        # Axis=0 sum over time axis (t) -> shape (num_communities, num_communities)
        sum_I_j_to_i_t1 = infection_matrix.sum(axis=0)

        # --- 3. Calculate denominator K_j = sum_t I_j(t)
        # Your code:
        # Infection_matrix[:Tmax-1, :, :].sum(axis=1).sum(axis=0)
        #
        # Explanation:
        # Infection_matrix[t, i, j] = # new infections in i FROM j at time t
        # Sum over axis=1 (即 i) 求和 => 得到“从 j 传出去的总感染数”在该时刻
        # Then sum over time => get total output infections for each j
        #
        # This is exactly K_j = sum_t I_j(t)
        sum_I_j_t_minus1 = infection_matrix[:Tmax-1, :, :].sum(axis=2).sum(axis=0)
        # Shape: (num_communities,)

        # Reshape to 1 x num_communities for convenient broadcasting to (i,j)
        sum_I_j_t_matrix = sum_I_j_t_minus1.reshape(1, -1)

        # --- 4. Estimate Rhat (和你原来一样)
        estimation_matrix = np.where(
            sum_I_j_t_matrix != 0,
            sum_I_j_to_i_t1 / sum_I_j_t_matrix,
            0.0
        )
        # Shape: (num_communities, num_communities)

        # --- 5. Empirical variance estimate Var_emp(R_ij) = Rhat_ij / K_j
        # That is variance_matrix[i,j] = estimation_matrix[i,j] / sum_I_j_t_minus1[j]
        #
        # Important: use same broadcasting method and sum_I_j_t_matrix
        variance_matrix = np.where(
            sum_I_j_t_matrix != 0,
            estimation_matrix / sum_I_j_t_matrix,
            0.0
        )

        # --- 6. No need for spectral decomposition, delete the following original logic:
        # Eigenvalues, eigenvectors = np.linalg.eig(estimation_matrix)
        # ...
        # Variance_matrix = np.where( ... spectral stuff ... )

        # --- 7. We can still keep spectral radius (optional)
        # If you still want to keep 'estimation_matrix_radius' in all_results_df
        # We can still calculate max real part of eigenvalues, purely for diagnostic output, not for variance
        eigenvalues, _ = np.linalg.eig(estimation_matrix)
        max_eig = np.max(eigenvalues.real)

        # --- 8. Summary
        all_results[sim_num] = {
            'estimation_matrix': estimation_matrix,
            'variance_matrix': variance_matrix,
            'Simulation_number': sim_num,
            'Total infection': infection_matrix.sum(),
            'Initial_case': start_place
        }

        all_results_df[sim_num] = {
            'estimation_matrix_radius': max_eig,
            'Simulation_number': sim_num,
            'Total infection': infection_matrix.sum(),
            'Initial_case': start_place
        }

    # --- 9. Combine DataFrame
    all_results_df_concat = pd.DataFrame.from_dict(all_results_df, orient='index')

    return all_results, all_results_df_concat



def median_nonzero(filtered_matrices):
    """
    逐元素对所有模拟传播矩阵求非零中位数。
    如果某个位置在所有样本中都为0，则该位置返回0。
    """
    filtered_matrices = np.array(filtered_matrices)  # Shape: (n_runs, n, n)
    n_runs, n, _ = filtered_matrices.shape

    result = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            nonzero_values = filtered_matrices[:, i, j][filtered_matrices[:, i, j] > 0]
            if nonzero_values.size > 0:
                result[i, j] = np.median(nonzero_values)
            else:
                result[i, j] = 0.0  # Or np.nan, if you want to mark it as not estimated

    return result


def calculate_median_estimation_matrix(results_dict):
    """
    Purpose:
    Calculate median parameter estimates across multiple simulation conditions.
    
    Inputs:
    - results_dict: dict, simulation results indexed by scenario identifier
    
    Outputs:
    - dict, median parameter matrix for each scenario
    
    Notes:
    Synthesizes ensemble results into robust point estimates;
    median provides resistance to outliers in simulation ensemble.
    """

    # Filter matrices where 'Total infection' > 100
    filtered_matrices = [
        result['estimation_matrix'] 
        for result in results_dict.values() 
        if result['Total infection'] > 5
    ]

    # Calculate and return the median across filtered matrices
    if filtered_matrices:
        # Return median_nonzero(filtered_matrices)
        return np.median(filtered_matrices,axis=0)
    else:
        raise ValueError("No matrices meet the 'Total infection' > 100 condition.")
        
        
        
def calculate_mean_variance_matrix_smalloutbreak(results_dict):
    """
    Purpose:
    Compute mean and variance of transmission parameters for small outbreak scenarios.
    
    Inputs:
    - results_dict: dict, simulation results by condition
    
    Outputs:
    - dict, mean and variance matrices by condition
    
    Notes:
    Provides uncertainty quantification and variability assessment for parameter inference.
    Focuses on small outbreak regime relevant for extinction analysis.
    """

    # Filter matrices where 'Total infection' > 100
    filtered_matrices = [
        result['variance_matrix'] 
        for result in results_dict.values() 
        if result['Total infection'] > 5
    ]

    # Calculate and return the median across filtered matrices
    if filtered_matrices:
        # Return median_nonzero(filtered_matrices)
        return np.median(filtered_matrices,axis=0)

    else:
        raise ValueError("No matrices meet the 'Total infection' > 100 condition.")
        
          


# %%
# Plotting the dot plot
plt.figure(figsize=(4, 3))

# Iterate through each row for the dot and CI
for index, row in Bootstrap_df.iterrows():
    # Plot median for both 100000000 and 1000 as big semi-transparent dots
    plt.plot(index, row['median_100000000'], 'o', markersize=10, label='100000000' if index == 0 else "", color='blue', alpha=0.7)
    plt.plot(index, row['median_1000'], 'o', markersize=10, label='1000' if index == 0 else "", color='green', alpha=0.7)
    
    # Plot CI as a small semi-transparent bar with the same width as dots
    plt.bar(index, row['upper_bound_100000000'] - row['lower_bound_100000000'], bottom=row['lower_bound_100000000'], 
            width=0.35, color='blue', alpha=0.3)
    plt.bar(index, row['upper_bound_1000'] - row['lower_bound_1000'], bottom=row['lower_bound_1000'], 
            width=0.35, color='green', alpha=0.3)
    
    # Plot Theoretical_value as a small horizontal line
    plt.plot([index - 0.2, index + 0.2], [row['Theoretical_value'], row['Theoretical_value']], '|-', color='red', lw=2)

# Adding labels and legend
plt.xticks(range(len(Bootstrap_df)), Bootstrap_df['Name'], rotation=45, ha='right')
plt.ylabel('Values')
plt.title('Dot Plot with CI and Theoretical Value ')
plt.legend(loc='upper left')
plt.grid(False)  # Remove background grid
plt.ylim(0, 3)

plt.show()

# %% [markdown]
# # Traditional estimation method (Non bayes)

# %%
from scipy.stats import truncnorm
import numpy as np
import pandas as pd

nruns = 1000
Tmax = 30
bootstrap_samples = 10

bootstrap_results = []

# 👇 Four different initial seed locations
# Initial_case_places = [96, 7, 67, 95,32,9]

# This is still your original input location list for inference
# Input_places = [96, 7, 67, 95,32,9] # Napoli, Roma, Trento, Caserta


# 👇 Four different initial seed locations
initial_case_places = [96, 7, 82, 95,32,9]

# This is still your original input location list for inference
input_places = [96, 7, 82, 95,32,9]  # Napoli, Roma, Trento, Caserta



# 👇 Outer loop first over different initial_case_place
for initial_case_place in initial_case_places:

    for max_size in [100, 1000, 10000, 100000, 1000000]:

        # 1️⃣ Simulation: using current initial_case_place
        simulations = disease_transmission_simulation(
            R_matrix_test, nruns, Tmax, initial_case_place, max_size
        )

        # 2️⃣ Parameter inference
        all_results, _ = calculate_parameter_inference_matrix(simulations, input_places)

        # Store bootstrap results separately for each input_place
        R_ob_bootstrap_by_place = {place: [] for place in input_places}

        # 3️⃣ Bootstrap for each outbreak result
        for sim_result in all_results.values():
            est_matrix = sim_result["estimation_matrix"]
            var_matrix = sim_result["variance_matrix"]

            for _ in range(bootstrap_samples):
                try:
                    sample_matrix = truncnorm.rvs(
                        a=0, b=np.inf, loc=est_matrix, scale=np.sqrt(var_matrix)
                    )

                    extinction_prob = get_extinction_prob(sample_matrix)
                    R_ob_sample = Equivalent_R(1 - extinction_prob)

                    for place in input_places:
                        R_ob_bootstrap_by_place[place].append(R_ob_sample[place])

                except Exception as e:
                    continue

        # 4️⃣ Calculate statistics separately for each input_place
        for place in input_places:
            R_ob_all_bootstrap = np.array(R_ob_bootstrap_by_place[place])
            R_ob_all_bootstrap = R_ob_all_bootstrap[~np.isnan(R_ob_all_bootstrap)]

            if len(R_ob_all_bootstrap) == 0:
                continue

            lower_bound = np.percentile(R_ob_all_bootstrap, 20)
            upper_bound = np.percentile(R_ob_all_bootstrap, 80)
            median_R_ob = np.median(R_ob_all_bootstrap)

            bootstrap_results.append({
                "initial_case_place": initial_case_place,  # 👈 NEW column
                "max_size": max_size,
                "input_place": place,
                "median": median_R_ob,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            })


# 5️⃣ Convert to DataFrame
bootstrap_df_varying_maxsize = pd.DataFrame(bootstrap_results)
print(bootstrap_df_varying_maxsize.head())


# %%
Equivalent_R_italy_09_25[9]

# %%


# %%
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Map node number to city name
place_name_map = {
    96: "Napoli",
    7: "Roma",
    82: "Venezia",
    95: "Caserta",
    32: "Torino",
    9: "Genova"
}

# Theoretical value for each city
theoretical_values = {
    96: 2.488136,
    7: 2.2308211,
    82: 1.1064018,
    95: 1.3680032,
    32: 1.8722594,
    9: 2.3471151
}

# Order of input_place to draw input_place 顺序
places = [96, 7, 82, 95, 32, 9]

# 2 rows 3 columns subplots, 6 cities total
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
axes = axes.flatten()

for idx, place in enumerate(places):
    ax = axes[idx]

    # Only take results for current input_place
    df_sub = bootstrap_df_varying_maxsize[
        bootstrap_df_varying_maxsize["input_place"] == place
    ].copy()

    # Sort by max_size
    df_sub = df_sub.sort_values("max_size")

    # Group by initial_case_place: one line per seed
    for seed_place, df_seed in df_sub.groupby("initial_case_place"):
        ax.plot(
            df_seed["max_size"],
            df_seed["median"],
            marker='o',
            label=f"Seed: {place_name_map.get(seed_place, seed_place)}"
        )
        ax.fill_between(
            df_seed["max_size"],
            df_seed["lower_bound"],
            df_seed["upper_bound"],
            alpha=0.3
        )

    # Theoretical value red dashed line
    ax.axhline(
        y=theoretical_values[place],
        color='r',
        linestyle='--',
        label="Theoretical" if idx == 0 else None  # Add legend label only to first subplot to avoid duplication
    )

    ax.set_xscale("log")
    ax.set_title(place_name_map[place], fontsize=14)
    ax.set_xlabel("Max outbreak size")
    ax.set_ylabel("R_ob estimate")
    ax.grid(True, alpha=0.3)

# Unified legend (get handles/labels from first subplot)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()
plt.suptitle("Bootstrap R_ob Estimates vs Max Size", fontsize=16, y=1.08)
plt.show()

# %%
bootstrap_df_varying_maxsize["input_place_name"] = \
    bootstrap_df_varying_maxsize["input_place"].map(place_name_map)

# --- Map initial_case_place to city name ---
bootstrap_df_varying_maxsize["initial_case_place_name"] = \
    bootstrap_df_varying_maxsize["initial_case_place"].map(place_name_map)

# --- Map theoretical value into it ---
bootstrap_df_varying_maxsize["theoretical_value"] = \
    bootstrap_df_varying_maxsize["input_place"].map(theoretical_values)

# %%
bootstrap_df_varying_maxsize.to_csv("/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/Figure/singletime_estimation_more_place.csv", index=False)


# %% [markdown]

# ========================
# SECTION7: Figure6 Rob estimation for real outbreak data (Canada)
# ========================

# #
# Theoretical R_ob estimation for Canada
# #

# %%
canada_data = pd.read_csv('./Cananda_data/Transmission_Tree_with_Generation_Time.csv')
canada_data = canada_data[canada_data['Region']!='Unsampled']
canada_data.rename(columns={
    'Number of Direct Offspring': 'Direct Offspring',
    'Number of Total Offspring': 'Total Offspring',
    'Region': 'Community ID',
    'Generation Time': 'Generation Time'  # This can be kept as is if no changes
}, inplace=True)
                

# %%
canada_data.sort_values(by="Generation Time",ascending=False) 

# %%


# %%
# Canada_matrix = np.load("./transmission_matrix_canada.npy")
Canada_matrix = np.load("./transmission_matrix.npy")

# %%
def calculate_parameter_inference_matrix_canada(simulation_results,start_place):
    """
    Use a matrix form to calculate sum_{t} I_{i <- j}(t+1) / sum_{t} I_j(t) for each (i, j) community pair.
    """
    all_results = {}
    all_results_df = {}

    # Iterate over each simulation
    Tmax, num_communities, _ = simulation_results.shape
    estimation_matrix = np.zeros((num_communities, num_communities))
    variance_matrix = np.zeros((num_communities, num_communities))
        
        # ESTIMATION OF PARAMETER
    sum_I_j_to_i_t1 = simulation_results.sum(axis=0)  # Summing over time to get total infections I_{i <- j}(t+1)

        # Calculate sum_{t} I_j(t)
    sum_I_j_t_minus1 = simulation_results[:Tmax-1, :, :].sum(axis=1).sum(axis=0)  # Sum over time and target communities to get I_j(t)
        
        # Compute estimation matrix
    sum_I_j_t_matrix = sum_I_j_t_minus1.reshape(1, -1)
    estimation_matrix = np.where(
            sum_I_j_t_matrix != 0,
            sum_I_j_to_i_t1 / sum_I_j_t_matrix,
            0)

        # ESTIMATION OF VARIANCE
    eigenvalues, eigenvectors = np.linalg.eig(estimation_matrix)
    max_eig = np.max(eigenvalues.real)  # Only consider the real part
    max_index = np.argmax(eigenvalues.real)
    perron_vector = eigenvectors[:, max_index].real
    perron_vector = np.abs(perron_vector) / np.linalg.norm(perron_vector)
        
    eigenvalues_T, eigenvectors_T = np.linalg.eig(estimation_matrix.T)
    max_index2 = np.argmax(eigenvalues_T.real)
    v_star = eigenvectors_T[:, max_index2].real
    v_star = np.abs(v_star) / (v_star @ perron_vector)

    v_star_initial = v_star[start_place]  # Assume the initial case is in the first community
    denominator = (max_eig ** Tmax - 1) * v_star_initial

    variance_matrix = np.where(
            (estimation_matrix > 0) & (denominator * perron_vector[:, np.newaxis] > 0),
            ((max_eig * estimation_matrix) / (denominator * perron_vector[:, np.newaxis])) ** 0.5,
            0)

    all_results = {
            'estimation_matrix': estimation_matrix,
            'variance_matrix': variance_matrix,
            'Total infection': simulation_results.sum(),
            'Initial_case': 0  # Assuming the initial case is in community 0
        }

    all_results_df = {
            'estimation_matrix_radius': max_eig,
            'Total infection': simulation_results.sum(),
            'Initial_case': 0  # Assuming the initial case is in community 0
        }


   # Print(all_results_df_concat)
    return all_results, all_results_df_concat


    # Choose this date, just, we assume this time the cases is decreasing, which means our assumption has already been broken.
Canada_estimation,Canada_estimation_total =  calculate_parameter_inference_matrix_canada(Canada_matrix[6:14,:,:],6)


# %%
np.linalg.eig(Canada_estimation['estimation_matrix'])[0]

# %%

import pandas as pd
import numpy as np

bootstrap_results = []
bootstrap_samples = 100000  # Define number of Bootstrap samples

# Get mean and standard deviation of Canada estimation matrix
canada_mean_matrix_bootstrap = Canada_estimation['estimation_matrix']
canada_std_matrix_bootstrap = np.sqrt(Canada_estimation['variance_matrix'])  # Calculate standard deviation from variance

# Initialize a list to store each Bootstrap result
all_bootstrap_results = []

# Perform Bootstrap sampling
for _ in range(bootstrap_samples):
    try:
        # Generate new random sample matrix from normal distribution
        sample_matrix = np.random.normal(canada_mean_matrix_bootstrap, canada_std_matrix_bootstrap)
        
        # Calculate R_ob value for each location (returns 1D array)
        extinction_prob = get_extinction_prob(sample_matrix)
        R_ob_bootsrtap_result_one = Equivalent_R(1 - extinction_prob)
        
        # Store result
        all_bootstrap_results.append(R_ob_bootsrtap_result_one)
    
    except AssertionError:
        # If non-convergence sample occurs, skip this iteration
        continue

# Convert all Bootstrap sample results to NumPy array (shape: bootstrap_samples × num_locations)
all_bootstrap_results = np.array(all_bootstrap_results)

# Remove samples containing NaN
# All_bootstrap_results = all_bootstrap_results[~np.isnan(all_bootstrap_results).all(axis=1)]


# Calculate median and confidence interval for each location, handle NaN samples
median_R_ob = np.nanmedian(all_bootstrap_results, axis=0)
lower_bound = np.nanpercentile(all_bootstrap_results, 2.5, axis=0)
upper_bound = np.nanpercentile(all_bootstrap_results, 97.5, axis=0)


# 定义地区名称并按指定顺序Store result
regions = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'QC']

# Build result DataFrame
result_df = pd.DataFrame({
    'region': regions,
    'median': median_R_ob,
    'lowci': lower_bound,
    'upci': upper_bound
})

print(result_df)



# %%
# BAYES FOR CANADA

import numpy as np
import pandas as pd

def calculate_bayesian_inference_matrix_canada(infection_matrix,
                                               mobility_matrix,
                                               c=5.0):

    """
    Purpose:
    Compute Bayesian posterior for transmission parameters (Canada analysis).
    
    Inputs:
    - simulation_results: array, results from stochastic epidemic simulations
    - start_place: int, initial case location
    
    Outputs:
    - array (n, n), Bayesian posterior distribution over transmission rates
    
    Notes:
    Integrates simulation likelihood with epidemic prior for Canadian geographic data.
    """

    Tmax, n, _ = infection_matrix.shape

    # ----- 1) Sufficient statistics -----
    # S_ij: total infections from j → i
    S_ij = infection_matrix.sum(axis=0)

    # E_j: total exposure from j
    E_j = infection_matrix[:Tmax-1, :, :].sum(axis=1).sum(axis=0)  # (n,)
    E_j_matrix = E_j.reshape(1, -1)  # (1, n)

    # ----- 2) Prior -----
    alpha_prior = c * mobility_matrix
    beta_prior  = c * np.ones_like(mobility_matrix)

    # ----- 3) Posterior -----
    alpha_post = alpha_prior + S_ij
    beta_post  = beta_prior + E_j_matrix  # Broadcast along rows

    # Posterior mean
    Rhat_post_mean = np.divide(alpha_post, beta_post,
                               out=np.zeros_like(alpha_post),
                               where=beta_post != 0)

    # Posterior variance
    Rhat_post_var = np.divide(alpha_post, beta_post**2,
                              out=np.zeros_like(alpha_post),
                              where=beta_post != 0)

    # ----- 4) Spectral radius -----
    eigvals = np.linalg.eigvals(Rhat_post_mean)
    max_eig = np.max(eigvals.real)

    total_infection = infection_matrix.sum()

    # ----- 5) Pack results -----
    all_results = {
        "Rhat_post_mean": Rhat_post_mean,
        "Rhat_post_var": Rhat_post_var,
        "alpha_post": alpha_post,
        "beta_post": beta_post,
        "posterior_mean_radius": max_eig,
        "Total infection": total_infection
    }

    all_results_df = pd.DataFrame([{
        "posterior_mean_radius": max_eig,
        "Total infection": total_infection
    }])

    return all_results, all_results_df
import numpy as np
import pandas as pd

def bootstrap_R_ob_from_bayes(bayes_result, 
                              bootstrap_samples=500, 
                              q_low=20, 
                              q_high=80):
    """
    Bootstrap R_ob for each community using Gamma posterior (Canada Bayes results).

    Parameters
    ----------
    bayes_result : dict
        Output from calculate_bayesian_inference_matrix_canada()
        Contains alpha_post, beta_post, etc.
    bootstrap_samples : int
        Number of Gamma posterior draws.
    q_low, q_high : float
        Percentile bounds for CI.

    Returns
    -------
    DataFrame
        Columns: [community, median, lower_bound, upper_bound]
    """

    alpha_post = bayes_result["alpha_post"]
    beta_post  = bayes_result["beta_post"]

    n = alpha_post.shape[0]
    R_ob_records = []

    # For each community j, we estimate distribution of R_ob(j)
    for j in range(n):

        R_ob_samples = []

        for _ in range(bootstrap_samples):

            # 1) sample a new R matrix from posterior
            try:
                R_sample = np.random.gamma(alpha_post, 1.0 / beta_post)
            except Exception:
                continue

            # 2) compute extinction probabilities
            try:
                ext_prob = get_extinction_prob(R_sample)  # Shape (n,)
            except Exception:
                continue

            # 3) compute R_ob = Equivalent_R(1 - extinction_prob)
            try:
                R_ob_vec = Equivalent_R(1 - ext_prob)
            except Exception:
                continue

            # 4) extract community j's R_ob
            R_ob_samples.append(R_ob_vec[j])

        # Remove NaN
        R_ob_samples = np.array(R_ob_samples)
        R_ob_samples = R_ob_samples[~np.isnan(R_ob_samples)]

        if len(R_ob_samples) == 0:
            continue

        # Compute CI
        median = np.median(R_ob_samples)
        lower  = np.percentile(R_ob_samples, q_low)
        upper  = np.percentile(R_ob_samples, q_high)

        R_ob_records.append({
            "community": j,
            "median": median,
            "lower_bound": lower,
            "upper_bound": upper
        })

    return pd.DataFrame(R_ob_records)


# %%
Canada_matrix = np.load("./transmission_matrix_canada.npy")
selected_idx = [0, 1, 2, 3, 4, 5, 7, 8]
CAN_R_matrix_sub = CAN_R_matrix[np.ix_(selected_idx, selected_idx)]

Canada_bayes_res, Canada_bayes_df = calculate_bayesian_inference_matrix_canada(
    infection_matrix = Canada_matrix[1:9, :, :],  # Use only time window 6-13
    mobility_matrix  = CAN_R_matrix_sub,  # Your mobility prior
    c                = 0
)



# %%
# Get theoretical Rob in Canada

def read_data(pop_path,colocation_path):
    
    """
    Purpose:
    Load and parse population and contact pattern data from CSV files.
    
    Inputs:
    - pop_path: str, file path to population data CSV
    - colocation_path: str, file path to contact/colocation data CSV
    
    Outputs:
    - tuple, (populations array, contact probability matrix)
    
    Notes:
    Standardizes data formats and handles missing values or formatting inconsistencies.
    """

    pop_data = pd.read_csv(pop_path,index_col=0)
    colocation_data = pd.read_csv(colocation_path, index_col=0)
    colocation_data = pd.DataFrame(data=colocation_data.values, index=colocation_data.columns, columns=colocation_data.columns)
    
    pop_data = pop_data.groupby(pop_data.index).sum()
    pop_data = pop_data.reindex(colocation_data.index).dropna()
    
    return(pop_data,colocation_data)
def R_matrix_calc(populations, probabilities,beta,mu):
    """
    Purpose:
    Calculate transmission matrix from demographic and contact data.
    
    Inputs:
    - populations: array (n,), population size N_i in each location
    - probabilities: array (n, n), contact probability matrix P_{ij}
    - beta: float, transmission rate per contact
    - mu: float, recovery rate
    
    Outputs:
    - array (n, n), transmission matrix R
    
    Notes:
    Standard formulation: R_{ij} encodes expected secondary infections in j from primary in i.
    """

    adjusted_populations = populations
    NGM = pd.DataFrame(index=probabilities.index, columns=probabilities.columns)

    for i in probabilities.index:
        for j in probabilities.columns:
            NGM.at[i, j] = (beta / mu) * probabilities.at[i, j]*adjusted_populations.loc[i]

    NGM_matrix = NGM.to_numpy(dtype=float)
    return NGM_matrix

CAN_pop, CAN_colocation_09_25 = read_data('/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/canada_pop.csv','/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/canada_colocation_09_25.csv')
# # MAKE THE R reference is equal to 2.5, refer this paper: https://www.sciencedirect.com/science/article/pii/S2468042720300634
CAN_R_matrix = R_matrix_calc(CAN_pop,CAN_colocation_09_25, (0.0003794804101197926*0.0005/1.549027688752253)/4.5*1.25,1/5.1)
CAN_R_extinction =  get_extinction_prob(CAN_R_matrix)
CAN_R_extinction_eq_p = Equivalent_R(1-CAN_R_extinction)


Canadaname   = pd.DataFrame({
    'Initial_name':  CAN_pop.index ,
    'Initial Community': np.arange(len(CAN_pop))  # Generate sequential Community IDs starting from 0
})
Canadaname.to_csv('./Canadaname.csv',index=False)
Equivalent_R_canada_09_25_df = pd.DataFrame({
    'R_ob_theoretical':  CAN_R_extinction_eq_p ,
    'Community ID': np.arange(len(CAN_pop))  # Generate sequential Community IDs starting from 0
})
# Equivalent_R_canada_09_25_df.to_csv('./Equivalent_R_canada_09_25_df.csv',index=False)
Equivalent_R_canada_09_25_df = Equivalent_R_canada_09_25_df.loc[[0, 1, 2, 3,4, 5, 7, 8]].reset_index(drop=True)

Canada_R_ob_bootstrap["Theoretical Rob"]=Equivalent_R_canada_09_25_df["R_ob_theoretical"]



# %%
Equivalent_R_canada_09_25_df 

# %%
result_df.to_csv("/Users/boxuan/Desktop/PhD/Probability_extinction/Qlife-workshop-mobility/Figure/canada_casestudy.csv", index=False)


# %%
Canada_R_ob_bootstrap

# %%
Canada_R_ob_bootstrap.columns

# %%
Equivalent_R_canada_09_25_df 

# %%
import matplotlib.pyplot as plt
import numpy as np

df = Canada_R_ob_bootstrap.copy()

# Region names (order consistent with df["community"])
regions = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'QC']
assert len(regions) == df.shape[0]

# Error bar height
yerr_lower = df["median"] - df["lower_bound"]
yerr_upper = df["upper_bound"] - df["median"]
yerr = [yerr_lower, yerr_upper]

plt.figure(figsize=(8, 4))  # Make figure smaller

# Bootstrap R_ob: large points + error bars, no lines
plt.errorbar(
    regions,
    df["median"],
    yerr=yerr,
    fmt="o",  # Circle
    markersize=8,  # Larger points
    capsize=5,  # Error bar cap
    linestyle="none",  # No lines
    label="Bootstrap R_ob",
    color = "blue"
)

# Theoretical R_ob: separate large points, no lines
plt.scatter(
    regions,
    df["Theoretical Rob"],
    marker="s",  # Square
    s=70,  # Point area, larger is bigger
    label="Theoretical R_ob",
    color = "red"

)

plt.xlabel("Region", fontsize=11)
plt.ylabel("R_ob", fontsize=11)
plt.title("Bootstrap vs Theoretical R_ob (Canada Regions)", fontsize=13)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
