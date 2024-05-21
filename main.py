import numpy as np 


F_k = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

R_k = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

Q_k = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

H_k = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
                         
B_k = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

U_k = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

Noise = np.array(
    [0.05, 0.05, 0.01, 0.01]
)

def getZ():
    print('get')

def predict(X_prev, P_prev):

    X_k = (F_k @ X_prev) + (B_k @ U_k)
    P_k = (F_k @ P_prev @ F_k.T) + (Q_k)

    return X_k, P_k 

def update(Z_k, X_prev, P_prev):
    Y_k = Z_k - (H_k @ X_prev)
    
    S_k = (H_k @ P_prev @ H_k.T) + (R_k)
    K_k = P_prev @ H_k.T @ np.linalg.pinv(S_k)
    X_k = X_prev + (K_k @ Y_k)
     
    P_k = P_prev - (K_k @ H_k @ P_prev)
     
    return X_k, P_k

def filter(X_prev, P_prev, Z_k):
    X_k, P_k = predict(X_prev, P_prev)
    print(f'State Estimate Before KF={X_k}')

    print(f'Observation={Z_k}')
             
    X_k, P_k = update(X_k, P_k)
    print(f'State Estimate After KF={X_k}') 

    return X_k, P_k

if __name__ == 'main':
    X_prev = np.array([0.0, 0.0, 0.0, 0.0])
    P_prev = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    Z_k = getZ()

    for i, k in enumerate(Z_k, start=1):
        print(f'\nState {k}:')
        X_k, P_k = filter(X_prev, P_prev)
        X_k = X_prev
        P_k = P_prev
        
