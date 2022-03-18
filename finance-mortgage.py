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

import numpy as np
import matplotlib.pyplot as plt


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
# -

# To convert yearly interest to monthly rate
#
# $ (1 + x)^{12} = (1 + I) $
#
# $ \log (1 + x) = \frac{1}{12} \log (1 + I) $
#
# $ x = (1 + I)^{1/12} - 1 $
#
# To find required monthly payment
#
# $ (T - m) $

# +
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
    
def calculate_payments(total, tenure, interest_year=0.02, downpayment_percent=0.25, grant=0):
    """
    Calculate remainining loan and interest for a given interest, tenure period,
    downpayment percentage and optional grant
    
    Parameters
    ----------
    total : Total cost
    interest_year : float (<1.0)
        Yearly interest rate (2% is 0.02)
    downpayment_period : float (<1.0)
        Percentage of total as downpayment
    grant : (optional) Amount of grant received, in addition to downpayment
    
    Returns
    -------
    remaining : ndarray
        (L,) array of the remaining loan
    interest : ndarray
        (L,) array of the interest / month
    
    """
    tenure_mths = 12 * tenure
    downpayment = total * downpayment_percent
    P = total - downpayment - grant
    interest_month = interest_yearly_to_monthly(interest_year)
    
    # Calculated required monthly payment to achieve tenure
    monthly = required_monthly_payment(P, tenure_mths, interest_month)

    t = np.arange(0, tenure_mths)
    remaining = np.zeros(tenure_mths)
    interest  = np.zeros(tenure_mths)
    remaining[0] = total - downpayment - grant

    for i in t[1:]:
        remaining[i] = remaining[i-1] - monthly
        interest[i]  = remaining[i-1] * interest_month
        remaining[i] += interest[i]
    
    # Clean up
    interest[0] = interest[1]
    remaining[remaining < 0] = np.nan
    interest[interest < 0] = np.nan
    
    return remaining, interest, tenure_mths, monthly

# total = 370
# grant = 50
# sol = calculate_payments(total, interest_year=0.02, tenure=25, grant=grant)
# remaining, interest, tenure_mths, monthly = sol

# total_interest = np.max(np.cumsum(interest)) * 1e3
# print(f"For a total of ${total}k, downpayment of ${total*0.25}k and grant of ${grant}k")
# print(f"Monthly: ${monthly * 1e3:.2f}")
# print(f"Total interest paid: ${total_interest :.0f}")
# print(f"Interest / month: ${total_interest / tenure_mths :.0f}")

# # %matplotlib inline
# plot_payments(tenure_mths, remaining, interest)


# -

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


# +
_,ax = plt.subplots(2,2, figsize=(16,8), dpi=300, sharex=False)
# plt.tight_layout()

total          = 370
tenure_range   = range(10,36,1)
inflation_rate = interest_yearly_to_monthly(0.02)

print(f"Plotting combinations for ${total}k\n")

remaining = []
interest_vs_tenure = []
interest_vs_tenure_infl = []
monthly_required = []
for i, grant in enumerate([0, 50]):
    R = np.zeros((12*35, len(tenure_range)))
    I_vs_T = []
    I_vs_T_infl = []
    m = []
    for j,tenure in enumerate(tenure_range):
        # Calculate P,I,T and M
        sol = calculate_payments(total, interest_year=0.02, tenure=tenure, grant=grant)
        P, interest, tenure_mths, monthly = sol
        
        t = tenure_mths/12
        R[:len(P),j] = P
        
        # Calculate interest amount relative to inflation
        inflation = np.power((1 + inflation_rate), np.arange(len(interest)))
        
        total_interest = np.max(np.cumsum(interest))
        total_interest_infl = np.max(np.cumsum(interest / inflation))
        
        # Store data
        I_vs_T.append( np.array([t, total_interest]) )
        I_vs_T_infl.append( np.array([t, total_interest_infl]) )
        m.append( np.array([t, monthly]) )
    
    # Aggregate data
    remaining.append( R )
    interest_vs_tenure.append( np.stack(I_vs_T) )
    interest_vs_tenure_infl.append( np.stack(I_vs_T_infl) )
    monthly_required.append( np.stack(m) )

# Create percentile fill_between for remaining over time
cmap = ['tab:blue', 'tab:orange']
remaining = np.nan_to_num(remaining)
remaining_percentile = []
t = np.arange(12*35)/12
for i,R in enumerate(remaining):
    for q in range(0,101,20):
        upper, lower = percentile_between(R, q)
        ax[i,0].fill_between(t, upper, lower, color=cmap[i], alpha=0.2) #, edgecolor=None)
        # ax[i,0].plot(t, upper, t, lower, color=cmap[i])
        if q == 0:
            ax[i,0].plot(t[upper > 0], upper[upper > 0], 'k--')

# Plot interest vs tenure period
for i, (I_vs_T, I_vs_T_infl) in enumerate(zip(interest_vs_tenure, interest_vs_tenure_infl)):
    ax[0,1].plot(I_vs_T[:,0], I_vs_T[:,1], '-x', color=cmap[i])
    ax[0,1].plot(I_vs_T[:,0], I_vs_T_infl[:,1], '--', color=cmap[i])
    # ax[0,1].plot(I_vs_T[:,0], inflation * I_vs_T[:,1], '--', color=cmap[i])

ax[0,1].legend(['25% down', '25% down + 2% inflation', '40% down', '40% down + 2% inflation'])

# Plot monthly payment vs tenure period
for m in monthly_required:
    ax[1,1].plot(m[:,0], m[:,1], '-x')

ax[1,1].legend(['25% down', '40% down'])

# Label
ax[0,0].set_ylabel("Remaining ($)")
ax[1,0].set_ylabel("Remaining (w/ Grant) ($)")
ax[0,1].set_ylabel("Total Interest ($)")
ax[1,1].set_ylabel("Monthly ($)")

ax[1,0].set_xlabel("time (years)")
ax[1,1].set_xlabel("tenure (years)")

_ = [ a.grid() for a in ax.flatten()  ]

# +
_, ax = plt.subplots(3, gridspec_kw={'height_ratios': [1,1,2]}, figsize=(12,16), dpi=300)
# plt.figure(figsize=(8,8), dpi=None)
cmap = ['tab:blue', 'tab:orange']


# =========================== Interest vs Tenure ===========================
for i, (I_vs_T, I_vs_T_infl) in enumerate(zip(interest_vs_tenure, interest_vs_tenure_infl)):
    ax[0].plot(I_vs_T[:,0], I_vs_T[:,1], '-x', color=cmap[i])
    ax[0].plot(I_vs_T[:,0], I_vs_T_infl[:,1], '--', color=cmap[i])
    # ax[0,1].plot(I_vs_T[:,0], inflation * I_vs_T[:,1], '--', color=cmap[i])

ax[0].legend(['25% down', '25% down + 2% inflation', '40% down', '40% down + 2% inflation'])

ax[0].set_ylabel("Total Interest ($)")
ax[0].set_xlabel("Tenure (years)")

# =========================== Monthly vs Tenure ===========================
for m in monthly_required:
    ax[1].plot(m[:,0], m[:,1], '-x')

ax[1].legend(['25% down', '40% down'])

ax[1].set_ylabel("Monthly ($)")
ax[1].set_xlabel("Tenure (years)")

# =========================== Monthly vs Total Interest ===========================
for i,_ in enumerate(interest_vs_tenure):
    I_vs_T = interest_vs_tenure[i]
    I_vs_T_infl = interest_vs_tenure_infl[i]
    m = monthly_required[i] * 1000
    
    # Momthly vs Total Interest
    ax[2].plot(m[:,1], I_vs_T[:,1], color=cmap[i])
    ax[2].plot(m[:,1], I_vs_T_infl[:,1], '--', color=cmap[i])

ax[2].legend(['25% down', '25% down + 2% inflation', '40% down', '40% down + 2% inflation'])

ax[2].set_ylabel("Total Interest ($k)")
ax[2].set_xlabel("Monthly Payment ($)")

_ = [ a.grid() for a in ax.flatten()  ]
# -

