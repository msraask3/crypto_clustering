import cvxpy as cp
import scipy.optimize as sco

def get_efficient_frontier(avg_returns, cov_mat):
    """
    Args:
        avg_returns (n_assets, )
        cov_mat (n_assets, n_assets)
    """
    n_assets = len(avg_returns)

    weights = cp.Variable(n_assets)
    gamma = cp.Parameter(nonneg=True)

    portf_rtn_cvx = avg_returns @ weights
    portf_vol_cvx = cp.quad_form(weights, cov_mat)

    objective_function = cp.Maximize(portf_rtn_cvx - gamma * portf_vol_cvx) # 목적함수
    constraints = [cp.sum(weights) == 1, weights >=0, weights <= 0.5] # 제약조건
    # constraints = [cp.sum(weights) == 1, weights >=0] # 제약조건

    problem = cp.Problem(objective_function, constraints) # 문제정의

    # Efficient frontier
    SAMPLES = 500
    portf_rtn_cvx_ef = np.zeros(SAMPLES)
    portf_vol_cvx_ef = np.zeros(SAMPLES)
    weights_ef = []
    # gamma_range = np.linspace(0, 1000, num=SAMPLES)
    gamma_range = np.logspace(-3, 5, num=SAMPLES)

    for i in range(SAMPLES):
        gamma.value = gamma_range[i]
        problem.solve()
        portf_rtn_cvx_ef[i] = portf_rtn_cvx.value
        portf_vol_cvx_ef[i] = cp.sqrt(portf_vol_cvx).value
        weights_ef.append(weights.value)
    
    return portf_rtn_cvx_ef, portf_vol_cvx_ef, weights_ef

# 음의 샤프지수 포트폴리오 함수
def neg_sharpe_ratio(w, avg_returns, cov_mat, rf_rate=0.):
    """
    Args:
        w (n_assets, )
        avg_returns (n_assets, )
        cov_mat (n_assets, n_assets)
        rf_rate (scalar): 무위험 수익률
    """
    portf_returns = np.sum(avg_returns * w)
    portf_volatility = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    portf_sharpe_ratio = (portf_returns - rf_rate) / portf_volatility

    return - portf_sharpe_ratio

# 포트폴리오 수익률 얻는 함수
def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns*w)

# 포트폴리오 변동성 얻는 함수
def get_portf_vol(w, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

# 최대 샤프지수 포트폴리오
def get_max_sharpe_portfolio(avg_returns, cov_mat, rf_rate=0.):
    n_assets = len(avg_returns)

    args = (avg_returns, cov_mat, rf_rate)
    constraints = (
        {
                "type": "eq",
                "fun": lambda x: np.sum(x) - 1
            }
        )

    bounds = tuple((0,0.5) for asset in range(n_assets))
    # bounds = tuple((0,1) for asset in range(n_assets))
    initial_guess = n_assets * [1. / n_assets]

    max_sharpe_portf = sco.minimize(
        fun = neg_sharpe_ratio,
        x0 = initial_guess,
        args = args,
        method = "SLSQP",
        bounds = bounds,
        constraints = constraints
    )

    max_sharpe_weights = max_sharpe_portf["x"]
    max_sharpe_returns = get_portf_rtn(max_sharpe_weights, avg_returns)
    max_sharpe_vol = get_portf_vol(max_sharpe_weights, cov_mat)
    max_sharpe_ratio = - max_sharpe_portf["fun"]

    max_sharpe_portfolio = {
        "Return": max_sharpe_returns.round(3), "Std": max_sharpe_vol.round(3),
        "Sharpe Ratio": max_sharpe_ratio.round(3), "Weights": max_sharpe_weights.round(3)
    }

    return max_sharpe_portfolio
