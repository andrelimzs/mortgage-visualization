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
    ax[0].set_ylabel("Remaining (k $)")
    
    ax[1].plot(t, np.cumsum(interest))
    ax[1].set_ylabel("Interest Paid (k $)")
    
    ax[2].plot(t, interest * 1e3)

    ax[2].set_xlabel("time (years)")
    _ = [ ax[i].grid() for i in range(3) ]

def interest_yearly_to_monthly(interest_year):
    """
    Convert a yearly interest rate to monthly
        x = (1 + interest_year) ^ {1/12} - 1
    """
    return (1 + interest_year) ** (1/12) - 1
    
def required_monthly_payment(P, T, I):
    """
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
    
    """
    A = 0
    for t in range(T-1):
        A += (1 + I)**t
    m = (1 + I)**T * P / A
    return m
    
def calculate_payments(price, tenure, interest_year=0.02, downpayment_percent=0.25, grant=0):
    """
    Calculate remainining loan and interest for a given interest, tenure period,
    downpayment percentage and optional grant
    
    Parameters
    ----------
    price : price cost
    interest_year : float (<1.0)
        Yearly interest rate (2% is 0.02)
    downpayment_period : float (<1.0)
        Percentage of price as downpayment
    grant : (optional) Amount of grant received, in addition to downpayment
    
    Returns
    -------
    remaining : ndarray
        (L,) array of the remaining loan
    interest : ndarray
        (L,) array of the interest / month
    
    """
    tenure_mths = 12 * tenure
    downpayment = price * downpayment_percent
    P = price - downpayment - grant
    interest_month = interest_yearly_to_monthly(interest_year)
    
    # Calculated required monthly payment to achieve tenure
    monthly = required_monthly_payment(P, tenure_mths, interest_month)

    t = np.arange(0, tenure_mths)
    remaining = np.zeros(tenure_mths)
    interest  = np.zeros(tenure_mths)
    remaining[0] = price - downpayment - grant

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
    """
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
    
    """
    upper = np.percentile(x, 50+q/2, axis=-1)
    lower = np.percentile(x, 50-q/2, axis=-1)
    return upper, lower


# + tags=[]
price_range    = [800, 600, 400]
tenure_range   = range(10,36,1)
tenure_nominal = tenure_range.index(25)
inflation_rate = interest_yearly_to_monthly(0.02)

cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# labels = [25, 37.5, 50]
labels = price_range

for i,_ in enumerate(price_range):
    print(f"Price: ${price_range[i]}k, Downpayment: ${0.25 * price_range[i]}k")
print(f"Tenure between {tenure_range[0]} to {tenure_range[-1]}")

# =========================== Calculate Monthly and Interest ===========================
remaining = []
interest = []
interest_vs_tenure = []
interest_vs_tenure_infl = []
monthly_required = []
for i, price in enumerate(price_range):
    R = np.zeros((12*35, len(tenure_range)))
    I = np.zeros((12*35, len(tenure_range)))
    IvT = []
    IvT_infl = []
    m = []
    for j,tenure in enumerate(tenure_range):
        # Calculate P,I,T and M
        sol = calculate_payments(price, interest_year=0.02, tenure=tenure, grant=0)
        P, Ik, tenure_mths, monthly = sol
        
        t = tenure_mths/12
        R[:len(P),j] = P
        I[:len(Ik),j] = Ik
        
        total_interest = np.max(np.cumsum(Ik))
        
        I_all_infl = []
        for infl in np.arange(0,4,1):
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
_,axes = plt.subplots(2,2, figsize=(16,9), dpi=100, sharex=False)
ax = axes.T.flatten()

remaining = np.nan_to_num(remaining)
t = np.arange(12*35)/12
for i,R in enumerate(remaining):
    R_nominal = R[:,tenure_nominal]
    ax[0].plot(t[R_nominal > 0], R_nominal[R_nominal > 0], 'k--')
    
    for T in [10,15,20,25,30,35]:
        j = tenure_range.index(T)
        ax[0].fill_between(t, R[:,j], R_nominal,
                           color=cmap[i], alpha=0.2, edgecolor='k')

ax[0].set_ylabel("Remaining ($)")
ax[0].set_xlabel("time (years)")

# =========================== Monthly vs Tenure ===========================
for i,m in enumerate(monthly_required):
    ax[1].plot(m[:,0], m[:,1] * 1000)

ax[1].set_ylabel("Monthly ($)")
ax[1].set_xlabel("Tenure (years)")

# =========================== Precentile Fill for Interest paid after N years ===========================
tYears = 10
tIndex = range(1 + 12*tYears)
tShort = t[tIndex]
for i,I in enumerate(interest):
    numMonths = np.arange( 1, len(I)+1, dtype=float )
    I_nominal = np.cumsum( 1000*I[:, tenure_nominal] )
    I_amortized_nominal = I_nominal / numMonths
    
    # Plot nominal lines
    ax[3].plot(tShort, I_nominal[tIndex], 'k--')
    ax[2].plot(tShort, I_nominal[tIndex] / numMonths[tIndex], 'k--')
    
    # Plot interest for different tenure periods
    for T in [10,15,20,25,30,35]:
        j = tenure_range.index(T)
        # Total interest (excluding payment toward the house itself)
        I_total = np.cumsum( I[:,j] * 1000, axis=0 )
        # Amortized interest over duration (months) owned
        I_amortized = I_total / numMonths
        
        ax[2].fill_between(tShort, I_amortized[tIndex], I_amortized_nominal[tIndex],
                           color=cmap[i], alpha=0.3, edgecolor='k')
        ax[3].fill_between(tShort, I_total[tIndex], I_nominal[tIndex],
                           color=cmap[i], alpha=0.3, edgecolor='k')
        
ax[2].set_ylabel("Interest / month ($)")
ax[3].set_ylabel("Total Interest ($)")
ax[2].set_xlabel("time (years)")
ax[2].set_ylim(100)

# =========================== Legend ===========================
patch = []
for i in range(len(remaining)):
    patch.append( mpatches.Patch(color=cmap[i], alpha=0.6, label=f'${labels[i]}k') )
    # orange_patch = mpatches.Patch(color='tab:orange', alpha=0.6, label='40% down')
ax[0].legend(handles=patch, loc="upper right")
ax[3].legend(handles=patch, loc="upper left")

_ = [ a.grid() for a in ax.flatten() ]

plt.tight_layout()
plt.savefig('./docs/plots/overview1.jpg', dpi=200)

# + tags=[]
_, ax = plt.subplots(2, gridspec_kw={'height_ratios': [1,2]}, figsize=(9,12), dpi=100)

# =========================== Interest vs Tenure ===========================
for i, (IvT, IvT_infl) in enumerate(zip(interest_vs_tenure, interest_vs_tenure_infl)):
    # Fill_between 0% and 4%
    ax[0].plot(IvT[:,0], IvT[:,1],
               color=cmap[i], label=f"${labels[i]}k")
    ax[0].plot(IvT[:,0], IvT_infl[:,1,-1], '--',
               color=cmap[i], label='_nolegend')
    for j in range(IvT_infl.shape[2]):
        ax[0].fill_between(IvT[:,0], IvT[:,1], IvT_infl[:,1,j], color=cmap[i], alpha=0.2, label='_nolegend_')

ax[0].legend()
# ax[0].legend(['25% down', '25% down + (1-4%) inflation', '40% down', '40% down + (1-4%) inflation'])

ax[0].set_ylabel("Total Interest ($)")
ax[0].set_xlabel("Tenure (years)")

# # =========================== Monthly vs Tenure ===========================
# for m in monthly_required:
#     ax[1].plot(m[:,0], m[:,1] * 1000, '-x')

# ax[1].legend(['25% down', '40% down'])

# ax[1].set_ylabel("Monthly ($)")
# ax[1].set_xlabel("Tenure (years)")

# =========================== Monthly vs Total Interest ===========================
for i,_ in enumerate(interest_vs_tenure):
    IvT = interest_vs_tenure[i]
    IvT_infl = interest_vs_tenure_infl[i]
    m = monthly_required[i] * 1000
    
    # Fill between 0% and 4% inflation
    ax[1].plot(m[:,1], IvT[:,1],
               color=cmap[i], label=f"${labels[i]}k")
    ax[1].plot(m[:,1], IvT_infl[:,1,-1], '--',
               color=cmap[i], label=f"${labels[i]}k + (1-4%) inflation")
    for j in range(IvT_infl.shape[2]):
        ax[1].fill_between(m[:,1], IvT[:,1], IvT_infl[:,1,j], color=cmap[i], alpha=0.2, label='_nolegend_')

# ax[1].legend(['25% down', '25% down + (1-4%) inflation', '40% down', '40% down + (1-4%) inflation'])
ax[1].legend()

ax[1].set_ylabel("Total Interest ($ k)")
ax[1].set_xlabel("Monthly Payment ($ k)")

_ = [ a.grid() for a in ax.flatten()  ]

plt.tight_layout()
plt.savefig('./docs/plots/overview2.jpg', dpi=200)
# -
