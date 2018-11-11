import numpy as np
import matplotlib.pyplot as plt

# Main file will be composed of several functions and subfunctions

# Functions
# (1) initialize(): Generate Win and W based on specified parameters:
#     Inputs: spectral radius (rho), sparsity of reservoir (alpha), size of reservoir (N) and dim. of time series (k)
#     Outputs: input to reservoir connectivity matrix (Win), reservoir connectivity matrix (W)
# (2) organize(): Organize data into format that program can use:
#     Inputs: set of time series to be classified (each time series is a matrix), set of associated teacher vectors
#     Outputs: matrix of teacher signals (D), list of time series (U)
# (3) train(): Function trains ESN
#     Inputs: Win, W, D, U, (see above for definitions), and f (activation fn)
#     Outputs: Wout (weight functions for reservoir to output layer)
# (4) TSC() --Time Series Classification: function classifies time series
#     Inputs: time series (u), Win, W, Wout, f (see above for definitions)
#     Outputs: Class of time series (y)
# (5) sigmoid(): Reservoir activation function (just a sigmoid function)
#     Inputs: value (x)
#     Outputs: output of sigmoid function. a is currently set to 0.1
# (6) test(): test model
#     Inputs: file name for where to find test data set (filename) and all inputs for
#     TSC except u
#     Outputs: error rate (ER)
# (7) fourier(): Fourier transform function
#     Inputs: function (x) and desired expansion length (R)
#     Outputs: Fourier coefficients (coeff)


# Function 5: sigmoid-----
# Reservoir activation function
def sigmoid(x):
    y = 1/(1+np.exp(-0.1*x))

    return y


# Function 6: fourier-----
def fourier(vec):
    T=len(vec)
    coeff=np.zeros(2*T-1)
    z=np.fft.fft(vec)
    alpha=z.real
    beta=z.imag
    coeff[0]=alpha[0]/np.sqrt(T)
    for k in range(1,T):
        coeff[2*k-1]=alpha[k]/np.sqrt(2*T)
        coeff[2*k]=-beta[k]/np.sqrt(2*T)
    return coeff


# NOTES: (1) Check if we should be inputing all time steps at once. Check if the distribution we are choosing weights
# from matters (e.g., normal or uniform or otherwise).
# Function 1: initialize-----
def initialize(rho=0.9,alpha=0.01,N=200,k=1,alpha_in=0.1,scale_Win=1/100):

    #Define W
    W=np.random.rand(N,N) #Uniformly distributed random matrix
    # W=np.random.rand(N, N)  # Standard normal weighting option

    Z=np.random.binomial(1,1-alpha,(N,N)) #Zero-one matrix with density 1-alpha (sparsity alpha)
    W=np.multiply(W,Z) #Element-wise multiply to get desired sparsity
    rhoW=np.max(np.absolute(np.linalg.eig(W)[0]))  # Find spectral radius of W
    W=rho*W/rhoW  # Scale spectral radius of W so that it is equal to rho

    #Define Win
    Win=np.random.rand(N,k)  # W has uniform random weights every input node connected to every reservoir node
    # Win=np.random.randn(N, k)  # Standard normal weighting option
    Z = np.random.binomial(1, 1-alpha_in, (N, k))  # Change sparsity of Win
    Win=np.multiply(Win,Z)  # (see above line)

    Win=scale_Win*np.max(W)*Win/np.max(Win)

    return Win, W
# Test: W[np.where(W!=0)]
#       np.sum(Win==0)/len(Win)

# Function 2: organize-----
# Import data. Defaults are for SwedishLeaf set (15 classes and 500 training time series)
def organize(numclass=15,filename='SwedishLeaf'):

    file=open('UCR_TS_Archive_2015/'+filename+'/'+filename+'_TRAIN','r')  # Load file
    U=list()  # Define list to hold time series
    D=np.zeros((1,numclass))  # Predefine first row of teacher matrix (this row will have to be deleted)

    for line in file:
        line=line.strip()  # Remove '\n' at end of line
        row=np.asarray(line.split(',')).astype(np.float)  # Set up row of u
        d=np.zeros((1, 15))  # Predefine teacher vector length
        d[0,int(row[0]-1)]=1  # Add one to location of class in teacher vector
        row=np.delete(row,0)  # Delete first entry (class number) from row
        U.append(row)  # Add row to list U
        D=np.concatenate((D,d),axis=0)  # Add new teacher vector to the bottom of teacher matrix

    D=np.delete(D,0,axis=0)  # Delete first row (of zeros) from teacher matrix
    file.close()  # Close file

    return U, D  # Return teacher matrix and TS train list


# UPDATE SO WE DON'T HAVE TO TAKE THE TRANSPOSE ON LINE 91
# Function 3: train-----
# Structure:
#       Loop through m:
#            (1) Generate Xm(t), the matrix whose rows are the functions x(i,t) for the ith reservoir neuron
#            (2) Fourier transform Xm(t) to get the matrix whose rows are the first R fourier coefficients of the
#                function x(i,t) for the mth sample
#            (3) Build the matrix 'M' whose rows are the R fourier coefficients of each time x(i,t), concatenated
#       Output matrix M
#       Calculate pseudoinverse of M, call it Mt
#       Multiply Mt*D to get Wout, the matrix of fourier coefficients of the reservoir output weight functions
def train(Win,W,D,U,f):
    m=D.shape[0]  # Get size of training data set 'm'
    N=W.shape[0]  # Get size of reservoir
    T=U[0].shape[0]  # Get length of first time series in list 'U' (every time series in U should be the same length)
    R=2*T-1  # Define number of coefficients of expansion (currently using all coefficients)
    M=np.zeros((m,N*R))  # .astype(complex)  # CHANGE THIS IF USING AMINE'S FUNCTION

    # Loop through list of training time series
    for i in range(0,m):

        X=np.zeros((N,T+1))  # .astype(complex)  # CHANGE THIS IF USING AMINE'S FUNCTION  # Predefine matrix to hold time series of reservoir activities
        Y=np.zeros((N,R)) #Define matrix to hold Fourier stuff

        for j in range(0,T):  # Loop to calculate reservoir time series
            X[:,j+1]=f(np.dot(W,X[:,j])+np.dot(Win,U[i][j]).reshape(N))  # UPDATE!!! time series #MAY HAVE TO UPDATE FOR MULTIDIM TS!!

        X = X[:,1:]  # Delete first column (of zeros)
        for j in range(0,N):  # Loop to calculate Fourier transform of reservoir time series (at each node)
            Y[j,:]=fourier(X[j,:])  # Calc. Fourier transform with Amine's function
            # X[j, :] = np.fft.fft(X[j, :], axis=0)  # Calc. Fourier transform with np.fft

        M[i,:]=Y.flatten(order='C')  # Define the ith row of M to be the row-major flattened matrix X

    Wout=np.dot(np.linalg.pinv(M),D)  # Calculate pseudoinverse of M and multiply by D for linear regression to get Wout

    return Wout


# Double check np.argmax (130)
# Function 4: TSC-----
# Structure:
#       (1) Generate X(t), the matrix whose rows are the functions x(i,t) for the ith reservoir neuron
#       (2) Fourier transform X(t) to get the matrix whose rows are the first R fourier coefficients of the
#           function x(i,t) for the mth sample
#       (3) Build the 1xNR vector 'M' that is the R fourier coefficients of each time x(i,t), concatenated
#       Output vector M
#       Multiply M*Wout to get Y
#       Find max value of Y. This is the class of the time series.
def TSC(u,W,Win,Wout,f):
    N=W.shape[0]  # Get size of reservoir
    T=u.shape[0]  # Get length of first time series u (MUST BE SAME LENGTH AS TRAINING TIME SERIES)
    R=2*T-1  # Define number of coefficients of expansion (currently using all coefficients)

    X = np.zeros((N, T+1))  # .astype(complex)  # Predefine matrix to hold time series of reservoir activities CHANGE IF USING AMINE'S F'N
    Y = np.zeros((N, R))  # Define matrix to hold Fourier stuff

    for j in range(0, T):  # Loop to calculate reservoir time series
        X[:, j + 1] = f(np.dot(W, X[:, j]) + np.dot(Win, u[j]).reshape(N))  # Update time series

    X = X[:, 1:]  # Delete first column (of zeros)
    for j in range(0, N):  # Loop to calculate Fourier transform of reservoir time series (at each node)
        Y[j, :] = fourier(X[j, :])  # Calc. Foureir transform using Amine's function
        # X[j, :] = np.fft.fft(X[j, :], axis=0)  # Calc. Foureir transform using np.fft.fft

    M = Y.flatten(order='C')  # Define vector M to be the row-major flattened matrix X

    Y=np.dot(M,Wout)  # Calculate proxy for output layer activities (don't understand why in Fourier domain?!?!?!?!)
    y=np.argmax(Y)+1  # Find location in output vector of max value as the max CHANGE IF USING AMINE'S F'N

    return y, Y  # y is class, as given by model, Y is output vector


# Function 6: test-----
# Tests data. Default dataset is SwedishLeaf
def test(W,Win,Wout,f,filename='SwedishLeaf'):

    # Organize data into a format that TSC can use
    file=open('UCR_TS_Archive_2015/'+filename+'/'+filename+'_TEST','r')  # Load file
    U=list()
    C=np.zeros((1,1))  # Predefine first row of class vector (this row will have to be deleted)

    for line in file:
        line=line.strip()  # Remove '\n' at end of line
        row=np.asarray(line.split(',')).astype(np.float)  # Set up row of u
        c=np.reshape(row[0],(1,1))  # c takes value of true class for the current time series
        row=np.delete(row,0)  # Delete first entry (class number) from row
        U.append(row)  # Add row to list U
        C=np.concatenate((C,c),axis=0)  # Add new class number to the bottom of class vector

    C=np.delete(C,0,axis=0)  # Delete first row (of zero) from class vector
    file.close()  # Close file

    # Run TSC on every time series in U
    y=np.zeros(len(U))  # Predefine output vector for classes given by model

    for i in range(0,len(U)):
        y[i]=TSC(U[i], W, Win, Wout, f)[0]

    ER=1-np.mean(np.equal(y,C.flatten(order='C'))*1)  # Compare TSC classifications (y) to true classes (C) element-wise and mean for error

    return ER, y, C




# Evaluate using SwedishLeaf
f = sigmoid  # Define f as the sigmoid function
Win,W = initialize(rho=0.9,alpha=0.01,N=200,k=1)  # Initialize first two matrices Win=W0[0], W=W0[1]
U,D = organize(numclass=15,filename='SwedishLeaf')  # Organize data: U=data[0], D=data[1]
Wout = train(Win,W,D,U,f)  # Train model
ER,y,C = test(W,Win,Wout,f,filename='SwedishLeaf')



