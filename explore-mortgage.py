# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python [conda env:control-systems] *
#     language: python
#     name: conda-env-control-systems-py
# ---

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# + [markdown] tags=[]
# ## Terminology
#
# - Price :  $P$
# Total price of a house
# - Tenure period : $T$ (years)
# Amount of time to fully pay off the mortgage
# - Monthly payment : $M$
# - Remaining value : $R[t]$
# - Interest : $I[t]$

# + tags=[]
def plot_payments(tenure_mths, remaining, interest):
    t = np.arange(tenure_mths) / 12
    _,ax = plt.subplots(3, figsize=(12,9), sharex=True)
    ax[0].plot(t, remaining)
    ax[0].set_ylabel('Remaining (k $)')
    
    ax[1].plot(t, np.cumsum(interest))
    ax[1].set_ylabel('Interest Paid (k $)')
    
    ax[2].plot(t, interest * 1e3)

    ax[2].set_xlabel('time (years)')
    _ = [ ax[i].grid() for i in range(3) ]

def interest_yearly_to_monthly(interest_year):
    '''
    Convert a yearly interest rate to monthly
        x = (1 + interest_year) ^ {1/12} - 1
    '''
    return (1 + interest_year) ** (1/12) - 1
    
def required_monthly_payment(P, T, I):
    '''
    Calculate required monthly payment for a given interest rate and tenure period
    
    Parameters
    ----------
    P : Total minus downpayment and grants
    T : Tenure, Number of months to fully pay off loan
    I : Monthly interest rate
    
    Returns
    -------
    monthly : Required monthly payment
    
    Notes
    -----
    Solution is given by the recurrence relation
        0 = ... (((P - m)(1 + I) - m)(1 + I) - m) ...
    
    Which can be unrolled into
        (1+I)^T P = (1 + I)^T m + (1 + I)^{T-1} m + ... + (1 + I)^0 m
    
    '''
    A = 0
    for t in range(T-1):
        A += (1 + I)**t
    m = (1 + I)**T * P / A
    return m
    
def calculate_payments(price, tenure, interest_year=0.02, downpayment_percent=0.25):
    '''
    Calculate remainining loan and interest for a given interest,
    tenure period, and downpayment percentage
    
    Parameters
    ----------
    price : ($)
    interest_year : float (<1.0)
        Yearly interest rate (2% is 0.02)
    downpayment_period : float (<1.0)
        Percentage of price as downpayment
    
    Returns
    -------
    remaining : ndarray
        (L,) array of the remaining loan
    interest : ndarray
        (L,) array of the interest / month
    
    '''
    tenure_mths = 12 * tenure
    downpayment = price * downpayment_percent
    P = price - downpayment
    interest_month = interest_yearly_to_monthly(interest_year)
    
    # Calculated required monthly payment to achieve tenure
    monthly = required_monthly_payment(P, tenure_mths, interest_month)

    t = np.arange(0, tenure_mths)
    remaining = np.zeros(tenure_mths)
    interest  = np.zeros(tenure_mths)
    remaining[0] = price - downpayment

    for i in t[1:]:
        remaining[i] = remaining[i-1] - monthly
        interest[i]  = remaining[i-1] * interest_month
        remaining[i] += interest[i]
    
    # Clean up
    interest[0] = interest[1]
    remaining[remaining < 0] = np.nan
    interest[interest < 0] = np.nan
    
    return remaining, interest, tenure_mths, monthly


# + tags=[]
def percentile_between(x, q):
    '''
    Return the q percentile of x about nominal value x0
    
    Parameters
    ----------
    x0 : ndarray
        (N,) array of Nominal values of x
    x : ndarray
        (N,K) array of Values
    q : float (0 < q < 100)
        Percentile
        
    Returns
    -------
    upper : ndarray
    lower : ndarray
    
    '''
    upper = np.percentile(x, 50+q/2, axis=-1)
    lower = np.percentile(x, 50-q/2, axis=-1)
    return upper, lower


# + tags=[]
price_range    = [800, 600, 400]
tenure_range   = range(10,36,1)
tenure_nominal = tenure_range.index(25)
inflation_rate = interest_yearly_to_monthly(0.02)
loan_interest  = 0.015
monthly_threshold = 3.6

cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# cmap = ['tab:green', 'tab:orange', 'tab:blue']
labels = price_range
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size']   = 14

print(f'Tenure between {tenure_range[0]} to {tenure_range[-1]}')
for i,_ in enumerate(price_range):
    print(f'Price: ${price_range[i]}k, Downpayment: ${0.25 * price_range[i]}k')

# =========================== Calculate Monthly and Interest ===========================
remaining = []
interest = []
interest_vs_tenure = []
interest_vs_tenure_infl = []
monthly_required = []
for i, price in enumerate(price_range):
    R = np.zeros((12*35, len(tenure_range)))
    I = np.zeros((12*35, len(tenure_range)))
    IvT, IvT_infl, m = [], [], []
    for j,tenure in enumerate(tenure_range):
        # Calculate P,I,T and M
        sol = calculate_payments(price, interest_year=loan_interest, tenure=tenure)
        P, Ik, tenure_mths, monthly = sol
        t = tenure_mths/12
        
        # Stop calculating when the monthly payment exceeds threshold
        if monthly > monthly_threshold:
            R[:,j] = np.nan
            I[:,j] = np.nan
            m.append( np.array([t, np.nan]) )
            continue
        
        R[:len(P),j] = P
        I[:len(Ik),j] = Ik
        
        total_interest = np.max(np.cumsum(Ik))
        
        I_all_infl = []
        for infl in np.arange(0,5,1):
            # Calculate interest amount relative to inflation
            inflation_rate = interest_yearly_to_monthly(infl/100)
            inflation_vector = np.power((1 + inflation_rate), np.arange(len(Ik)))    
            total_interest_infl = np.max(np.cumsum(Ik / inflation_vector))
            I_all_infl.append( np.array([t, total_interest_infl]) )
        
        # Store data
        IvT.append( np.array([t, total_interest]) )
        IvT_infl.append( np.stack(I_all_infl, axis=-1) )
        m.append( np.array([t, monthly]) )
    
    # Aggregate data
    remaining.append( R )
    interest.append( I )
    interest_vs_tenure.append( np.stack(IvT) )
    interest_vs_tenure_infl.append( np.stack(IvT_infl) )
    monthly_required.append( np.stack(m) )

# =========================== Precentile Fill for Remaining over time ===========================
_,axes = plt.subplots(2,2, figsize=(16,9), dpi=200, sharex=False)
ax = axes.T.flatten()

remaining = np.nan_to_num(remaining)
t = np.arange(12*35)/12
for i,R in enumerate(remaining):
    R_nominal = R[:,tenure_nominal]
    
    # Filter R_nominal > 0 so that the line stops when it hits the x-axis
    ax[0].plot(t[R_nominal > 0], R_nominal[R_nominal > 0], 'k--')
    
    for T in [10,15,20,25,30,35]:
        j = tenure_range.index(T)
        
        # If this tenure results in a monthly > threshold, skip it
        if np.sum(R[:,j]) == 0:
            continue
        
        ax[0].fill_between(t, R[:,j], R_nominal,
                           color=cmap[i], alpha=0.3, edgecolor='k')

ax[0].set_ylabel('Principal ($)')
ax[0].set_xlabel('time (years)')

# =========================== Monthly vs Tenure ===========================
for i,m in enumerate(monthly_required):
    ax[1].plot(m[:,0], m[:,1] * 1000, linewidth=2)

ax[1].set_ylabel('Repayment ($)')
ax[1].set_xlabel('Tenure (years)')

# =========================== Precentile Fill for Interest paid after N years ===========================
tYears = 10
tIndex = range(1 + 12*tYears)
tShort = t[tIndex]
for i,I in enumerate(interest):
    numMonths = np.arange( 1, len(I)+1, dtype=float )
    I_nominal = np.cumsum( I[:, tenure_nominal] )
    I_amortized_nominal = I_nominal / numMonths * 1000
    
    # Plot nominal lines
    ax[3].plot(tShort, I_nominal[tIndex], 'k--')
    ax[2].plot(tShort, I_nominal[tIndex] / numMonths[tIndex], 'k--')
    
    # Plot interest for different tenure periods
    for T in [10,15,20,25,30,35]:
        j = tenure_range.index(T)
        
        # Total interest (excluding payment toward the house itself)
        I_total = np.cumsum( I[:,j], axis=0 )
        
        # Amortized interest over duration (months) owned
        I_amortized = I_total / numMonths * 1000
        
        ax[2].fill_between(tShort, I_amortized[tIndex], I_amortized_nominal[tIndex],
                           color=cmap[i], alpha=0.3, edgecolor='k')
        ax[3].fill_between(tShort, I_total[tIndex], I_nominal[tIndex],
                           color=cmap[i], alpha=0.3, edgecolor='k')
        
ax[2].set_ylabel('Repayment to Interest / month ($)')
ax[3].set_ylabel('Accrued Interest ($)')
ax[2].set_xlabel('time (years)')
ax[3].set_xlabel('time (years)')
ax[2].set_ylim(0)

# =========================== Legend ===========================
patch = []
for i in range(len(remaining)):
    patch.append( mpatches.Patch(color=cmap[i], alpha=0.6, label=f'${labels[i]}k') )
ax[0].legend(handles=patch, loc='upper right')
ax[3].legend(handles=patch, loc='upper left')

_ = [ a.grid() for a in ax.flatten() ]
plt.tight_layout()
plt.savefig('./docs/plots/overview1.jpg')

# + tags=[]
_, axes = plt.subplots(1,2, figsize=(16,8), dpi=200)
ax = axes.flatten()

# =================== Interest vs Tenure (For variable duration) ===================
total_accrued_interest = []
for i, I in enumerate(interest):
    total_accrued_interest.append( np.cumsum(I, axis=0) )

min_tenure = 2
max_tenure = 14
t = np.array(tenure_range)
for i, accrued in enumerate(total_accrued_interest):
    ax[0].plot(t, accrued[12*min_tenure,:] / min_tenure, 'k--')
    for k, years in enumerate(range(min_tenure, max_tenure+1, 2)):
        j = 12 * years
        ax[0].fill_between(t, accrued[j,:] / years,
                           accrued[12*min_tenure,:] / min_tenure,
                           color=cmap[i], alpha=0.2, edgecolor='k',
                           label='_nolegend_')
    
ax[0].set_ylabel('Accrued Interest / year ($)')
ax[0].set_xlabel('Tenure (years)')

# =========================== Repayment vs Accrued Interest ===========================
for i, accrued in enumerate(total_accrued_interest):
    m = monthly_required[i]
    ax[1].plot(m[:,1], accrued[12*min_tenure,:], 'k--')
    for k, years in enumerate(range(min_tenure, max_tenure+1, 2)):
        j = 12 * years
        ax[1].plot(m[:,1], accrued[j,:], color=cmap[i], alpha=0.3)
        ax[1].fill_between(m[:,1], accrued[j,:],
                           accrued[12*min_tenure,:],
                           color=cmap[i], alpha=0.2, edgecolor='k',
                           label='_nolegend_')

ax[1].set_ylabel('Accrued Interest ($)')
ax[1].set_xlabel('Repayment ($)')

_ = [ a.grid() for a in ax.flatten()  ]
plt.tight_layout()
plt.savefig('./docs/plots/overview2.jpg')
# + jupyter={"source_hidden": true} tags=[]
# # =================== Interest vs Tenure (Assume full duration) ===================
# for i, (IvT, IvT_infl) in enumerate(zip(interest_vs_tenure, interest_vs_tenure_infl)):
#     # Fill_between 0% and 4%
#     ax[0].plot(IvT[:,0], IvT[:,1],
#                color=cmap[i],
#                label=f'${labels[i]}k')
    
#     ax[0].plot(IvT[:,0], IvT_infl[:,1,-1], '--',
#                color=cmap[i],
#                label='_nolegend_')
    
#     for j in range(IvT_infl.shape[2]):
#         ax[0].fill_between(IvT[:,0], IvT[:,1], IvT_infl[:,1,j],
#                            color=cmap[i], alpha=0.2, edgecolor='k',
#                            label='_nolegend_')

# ax[0].legend()
# ax[0].set_ylabel('accrued Interest ($)')
# ax[0].set_xlabel('Tenure (years)')

# # =========================== Repayment vs Accrued Interest ===========================
# for i, accrued in enumerate(total_accrued_interest):
#     m = monthly_required[i]
#     ax[1].plot(m[:,1], accrued[12*min_tenure,:], 'k--')
#     for k, years in enumerate(range(min_tenure, max_tenure+1, 2)):
#         j = 12 * years
#         ax[1].plot(m[:,1], accrued[j,:], color=cmap[i], alpha=0.3)
#         ax[1].fill_between(m[:,1], accrued[j,:],
#                            accrued[12*min_tenure,:],
#                            color=cmap[i], alpha=0.2, edgecolor='k',
#                            label='_nolegend_')

# ax[1].invert_xaxis()
# ax[1].set_ylabel('Accrued Interest ($)')
# ax[1].set_xlabel('Repayment ($)')
