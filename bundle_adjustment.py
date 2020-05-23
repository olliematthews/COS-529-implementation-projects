'''
An implementation of bundle adjustment. We reparameterise rotations in terms of 
Eular angles in order to ensure rotation matrices are valid. We use Levenberg
Marquardt optimisation. 
Sorry about all the yeboiiis - I ran out of variable names and creativity.
'''
import numpy as np
import pickle
import argparse


def forward_pass(P, X, f):
    # Runs a foward pass through the model
    X_dash = P @ X.T
    x_dash = f * X_dash[0:2, :] / X_dash[2, :]
    return X_dash, x_dash.T

def get_error(solution, n_cameras, n_points_tot):
    # Get the norm of the residuals
    residuals = []

    [residuals.append(np.zeros([2 * n_points_tot,])) for i in range(n_cameras)]
    for camera in range(n_cameras):
        points_present = [b[1] for b in solution['observations'] if b[0] == camera]
        n_points = len(points_present)
        eqs_present = (np.repeat([[0,1]], n_points, axis = 0) + np.array(points_present)[:,None] * 2).ravel()
        X = np.append(solution['points'][points_present,:], np.ones([n_points,1]), axis = 1)

        P = solution['poses'][camera,:3,:]
        f = solution['focal_lengths'][camera]

        X_dash, x_dash = forward_pass(P, X, f)
        x = np.array([b[2:4] for b in solution['observations'] if b[0] == camera])
        residuals[camera][eqs_present] = np.ravel(x - x_dash)
        
    big_res = np.zeros([0,])
    for res in residuals:
        big_res = np.append(big_res, res)
    return np.linalg.norm(big_res)


def get_AB(n_cameras, n_points_tot, solution):
    # Get the step lengths for the case where f is known
    
    # Initialise
    A = []
    [A.append(np.zeros([2 * n_points_tot, 6])) for i in range(n_cameras)]
    B = []
    for i in range(n_cameras):
        B_sub = []
        for i in range(n_points_tot):
            [B_sub.append(np.zeros([2, 3]))]
        B.append(B_sub)
        
    residuals = []
    [residuals.append(np.zeros([2 * n_points_tot,])) for i in range(n_cameras)]
    
    # Loop through each camera
    for camera in range(n_cameras):
        points_present = [b[1] for b in solution['observations'] if b[0] == camera]
        n_points = len(points_present)
        eqs_present = (np.repeat([[0,1]], n_points, axis = 0) + np.array(points_present)[:,None] * 2).ravel()
        X = np.append(solution['points'][points_present,:], np.ones([n_points,1]), axis = 1)

        P = solution['poses'][camera,:3,:]
        angles = solution['angles'][camera, :]
        f = solution['focal_lengths'][camera]
        X_dash, x_dash = forward_pass(P, X, f)
        x = np.array([b[2:4] for b in solution['observations'] if b[0] == camera])
        residuals[camera][eqs_present] = np.ravel(x - x_dash)
        
        # Get angle gradients
        R_deriv_0, R_deriv_1, R_deriv_2 = get_derivs(angles)
    
        # Calculate A
        boi = X / X_dash[2,:][:,None] * f 
        angular_boi_x = boi[:,:3] @ R_deriv_0 
        angular_boi_y = boi[:,:3] @ R_deriv_1
        yeboiii = - (x_dash[:,:,None] * X[:,None,:] / X_dash[2,:][:,None,None]).reshape([2 * n_points,-1])
        angular_yeboiii = yeboiii[:,:3] @ R_deriv_2
        total_yeboiii = np.append(angular_yeboiii, np.zeros([yeboiii.shape[0], 3]), axis = 1)
        total_yeboiii[np.arange(0, yeboiii.shape[0], 2),3] = boi[:,3]
        total_yeboiii[np.arange(0, yeboiii.shape[0], 2),:3] += angular_boi_x
        total_yeboiii[np.arange(1, yeboiii.shape[0], 2),4] = boi[:,3]
        total_yeboiii[np.arange(1, yeboiii.shape[0], 2),:3] += angular_boi_y
        total_yeboiii[:,5] = yeboiii[:,3]
        
        A[camera][eqs_present] = total_yeboiii

        # Calculate B        
        boi_1 = f * (P[None,0,:3] / X_dash[2][:,None])
        boi_2 = f * (P[None,1,:3] / X_dash[2][:,None])
        boi_3 = -(P[None,None,2,:3] * (x_dash / X_dash[2][:,None])[:,:,None])
        
        for i, point in enumerate(points_present):
            B[camera][point][0] = boi_1[i]
            B[camera][point][1] = boi_2[i]
            B[camera][point] += boi_3[i]
        
    return A, B, residuals



def get_AB_wf(n_cameras, n_points_tot, solution):
    # Get the step lengths for the case where f is not known
    
    # Initialise

    A = []
    [A.append(np.zeros([2 * n_points_tot, 7])) for i in range(n_cameras)]
    B = []
    for i in range(n_cameras):
        B_sub = []
        for i in range(n_points_tot):
            [B_sub.append(np.zeros([2, 3]))]
        B.append(B_sub)
    residuals = []
    [residuals.append(np.zeros([2 * n_points_tot,])) for i in range(n_cameras)]
    
    # Loop through the cameras
    for camera in range(n_cameras):
        points_present = [b[1] for b in solution['observations'] if b[0] == camera]
        n_points = len(points_present)
        eqs_present = (np.repeat([[0,1]], n_points, axis = 0) + np.array(points_present)[:,None] * 2).ravel()
        X = np.append(solution['points'][points_present,:], np.ones([n_points,1]), axis = 1)

        P = solution['poses'][camera,:3,:]
        angles = solution['angles'][camera, :]
        f = solution['focal_lengths'][camera]
        X_dash, x_dash = forward_pass(P, X, f)
        x = np.array([b[2:4] for b in solution['observations'] if b[0] == camera])
        residuals[camera][eqs_present] = np.ravel(x - x_dash)
        
        # Get angle gradients
        R_deriv_0, R_deriv_1, R_deriv_2 = get_derivs(angles)
    
        # Calculate A
        boi = X / X_dash[2,:][:,None] * f 
        angular_boi_x = boi[:,:3] @ R_deriv_0 
        angular_boi_y = boi[:,:3] @ R_deriv_1
        yeboiii = - (x_dash[:,:,None] * X[:,None,:] / X_dash[2,:][:,None,None]).reshape([2 * n_points,-1])
        angular_yeboiii = yeboiii[:,:3] @ R_deriv_2
        fin_boi = x_dash.ravel() / f
        total_yeboiii = np.append(angular_yeboiii, np.zeros([yeboiii.shape[0], 4]), axis = 1)
        total_yeboiii[np.arange(0, yeboiii.shape[0], 2),3] = boi[:,3]
        total_yeboiii[np.arange(0, yeboiii.shape[0], 2),:3] += angular_boi_x
        total_yeboiii[np.arange(1, yeboiii.shape[0], 2),4] = boi[:,3]
        total_yeboiii[np.arange(1, yeboiii.shape[0], 2),:3] += angular_boi_y
        total_yeboiii[:,5] = yeboiii[:,3]
        total_yeboiii[:,6] = fin_boi

        A[camera][eqs_present] = total_yeboiii

        # Calculate B        
        boi_1 = f * (P[None,0,:3] / X_dash[2][:,None])
        boi_2 = f * (P[None,1,:3] / X_dash[2][:,None])
        boi_3 = -(P[None,None,2,:3] * (x_dash / X_dash[2][:,None])[:,:,None])
        
        for i, point in enumerate(points_present):
            B[camera][point][0] = boi_1[i]
            B[camera][point][1] = boi_2[i]
            B[camera][point] += boi_3[i]
            
    return A, B, residuals


def get_UVW(A, B, residuals, n_points, n_cameras, is_calibrated):
    # Get U, W, V, and other needed matrices, leveraging sparsity where possible

    n_params = 6 if is_calibrated else 7
    U = np.zeros([n_params * n_cameras, n_params * n_cameras])
    res_A = []
    for i, (A_sub, res_sub) in enumerate(zip(A, residuals)):
        U[i * n_params : (i + 1) * n_params, i * n_params : (i + 1) * n_params] = A_sub.T @ A_sub
        res_A.append(A_sub.T @ res_sub)
    res_A = np.ravel(res_A)
    
   
    W = []
    res_B = []
    for i in range(n_points):
        W.append(np.zeros([0,3]))
        res_B.append(np.zeros([3,]))

    
    for A_sub, B_sub, res in zip(A, B, residuals):
        A_sub_reshape = A_sub.reshape([n_points,2,n_params])
        res_reshape = res.reshape([n_points, 2])
        for i, (A_subsub, B_subsub, res_subsub) in enumerate(zip(A_sub_reshape, B_sub, res_reshape)):
            W[i] = np.append(W[i], A_subsub.T @ B_subsub, axis = 0)
            res_B[i] += np.dot(B_subsub.T, res_subsub)


    V = []
    [V.append(np.zeros([3,3])) for i in range(n_points)]
    for B_sub in B:
        for i, B_sub_sub in enumerate(B_sub):
            V_sub = B_sub_sub.T @ B_sub_sub
            V[i] += V_sub
    WVi = []
    [WVi.append((np.linalg.solve(V_sub.T, W_sub.T)).T) for V_sub, W_sub in zip(V,W)]

    WViW = np.zeros([WVi[0].shape[0],WVi[0].shape[0]])
    WVires = np.zeros([WVi[0].shape[0],])
    for WVi_sub, W_sub, res_sub in zip(WVi, W, res_B):
        WViW += WVi_sub @ W_sub.T
        WVires += WVi_sub @ res_sub
    
    return U, V, W, WViW, WVires, res_A, res_B


def get_steps(U, V, W, WViW, WVires, res_A, res_B, lam):
    # Get the steps for the parameters for a given lambda
    
    
    # Account for the lambda factors
    U_star = U * (1 + lam)
    V_star = [V_sub * (1 + lam) for V_sub in V]
    WViW_star = WViW / (1 + lam)
    WVires_star = WVires / (1 + lam)

    # Get step lengths
    delta_A = np.linalg.solve(U_star.T - WViW_star,res_A - WVires_star)
    delta_B = np.zeros([0,])
    for i, (W_sub, V_sub, res_sub) in enumerate(zip(W, V_star, res_B)):
        delta_B = np.append(delta_B, np.linalg.solve(V_sub, res_sub - W_sub.T @ delta_A))
    return delta_A, delta_B



def get_rot(angles):
    # Get the rotation matrix for a set of angles
    ca, cb, cg = np.cos(angles)
    sa, sb, sg = np.sin(angles)
    R = np.array([[ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
              [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
              [-sb, cb * sg, cb * cg]])
    return R

def get_angles(R):
    # Get the angles for a rotation matrix
    a = np.arctan2(R[1,0], R[0,0])
    b = np.arctan2(-R[2,0] * np.cos(a),  R[0,0])
    g = np.arctan2(R[2,1], R[2,2])
    return np.array([a,b,g])

def get_derivs(angles):
    # Get the angular derivatives wrt the pose matrix rows
    ca, cb, cg = np.cos(angles)
    sa, sb, sg = np.sin(angles)
            
    R_deriv_0 = np.array([[- sa * cb, - sa * sb * sg - ca * cg, - sa * sb * cg + ca * sg],
              [- ca * sb, ca * cb * sg, ca * cb * cg],
              [0, ca * sb * cg + sa * sg, - ca * sb * sg + sa * cg]]).T
    
    
    R_deriv_1 = np.array([[ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
              [- sa * sb, sa * cb * sg, sa * cb * cg],
              [0, sa * sb * cg - ca * sg, -sa * sb * sg - ca * cg]]).T


    R_deriv_2 = np.array([[0, 0, 0],
              [-cb, - sb * sg, - sb * cg],
              [0, cb * cg, - cb * sg]]).T

    return R_deriv_0, R_deriv_1, R_deriv_2


def step(is_calibrated, solution, n_cameras, n_points_tot, lam, prev_error, stopping_conditions):
    # Do one gradient step
    if solution['is_calibrated']:
        A, B, residuals = get_AB(n_cameras, n_points_tot, solution)
        U, V, W, WViW, WVires, res_A, res_B = get_UVW(A, B, residuals, n_points_tot, n_cameras, True)
        for i in range(stopping_conditions[1]):
            delta_A, delta_B = get_steps(U, V, W, WViW, WVires, res_A, res_B, lam)

            solution_estimate = solution.copy()

            P_delta = delta_A.reshape([-1,6])
            solution_estimate['angles'] += P_delta[:,:3]
            # Keep angles in range
            solution_estimate['angles'] %= 2 * np.pi
            translations = solution['poses'][:,:3,3] + P_delta[:,3:]
            Rs = np.array([get_rot(a) for a in solution_estimate['angles']])
            
            solution_estimate['poses'][:,:3,:3] = Rs
            solution_estimate['poses'][:,:3,3] = translations
            
            X_delta = delta_B.reshape([-1,3])
            solution_estimate['points'] += X_delta
            
            error = get_error(solution_estimate, n_cameras, n_points_tot)
            print(f'Residual norm = {error}')
            if error >= prev_error:
                lam *= 10
            else:
                flag = (np.linalg.norm(np.append(delta_A, delta_B)) <= stopping_conditions[0])
                lam /= 10
                return solution_estimate, error, flag, lam
        return solution, error, True, lam
    
    
    else:
        A, B, residuals = get_AB_wf(n_cameras, n_points_tot, solution)
        U, V, W, WViW, WVires, res_A, res_B = get_UVW(A, B, residuals, n_points_tot, n_cameras, False)
        for i in range(stopping_conditions[1]):
            delta_A, delta_B = get_steps(U, V, W, WViW, WVires, res_A, res_B, lam)

            solution_estimate = solution.copy()

            P_delta = delta_A.reshape([-1,7])
            solution_estimate['angles'] += P_delta[:,:3]
            # Keep angles in range
            solution_estimate['angles'] %= 2 * np.pi
            translations = solution['poses'][:,:3,3] + P_delta[:,3:6]
            solution_estimate['focal_lengths'] += P_delta[:,6]
            Rs = np.array([get_rot(a) for a in solution_estimate['angles']])
            
            solution_estimate['poses'][:,:3,:3] = Rs
            solution_estimate['poses'][:,:3,3] = translations
            
            X_delta = delta_B.reshape([-1,3])
            solution_estimate['points'] += X_delta
            error = get_error(solution_estimate, n_cameras, n_points_tot)
            print(f'Residual norm = {error}')
            if error >= prev_error:
                lam *= 10
            else:
                flag = (np.linalg.norm(np.append(delta_A, delta_B)) <= stopping_conditions[0])
                lam /= 10
                return solution_estimate, error, flag, lam
        return solution, error, True, lam

def solve_ba_problem(problem):
    '''
    Solves the bundle adjustment problem defined by "problem" dict

    Input:
        problem: bundle adjustment problem containing the following fields:
            - is_calibrated: boolean, whether or not the problem is calibrated
            - observations: list of (cam_id, point_id, x, y)
            - points: [n_points,3] numpy array of 3d points
            - poses: [n_cameras,3,4] numpy array of camera extrinsics
            - focal_lengths: [n_cameras] numpy array of focal lengths
    Output:
        solution: dictionary containing the problem, with the following fields updated
            - poses: [n_cameras,3,4] numpy array of optmized camera extrinsics
            - points: [n_points,3] numpy array of optimized 3d points
            - (if is_calibrated==False) then focal lengths should be optimized too
                focal_lengths: [n_cameras] numpy array with optimized focal focal_lengths

    Your implementation should optimize over the following variables to minimize reprojection error
        - problem['poses']
        - problem['points']
        - problem['focal_lengths']: if (is_calibrated==False)

    '''
    solution = problem
        
    n_cameras = problem['poses'].shape[0]
    n_points_tot = problem['points'].shape[0]
    
    angles = np.zeros([0, 3])
    for pose in problem['poses']:
        angs = get_angles(pose[:3,:3]).reshape([1,-1])
        angles = np.append(angles, angs, axis = 0)
        
    solution.update({'angles' : angles})
    
    # Initialise variables 
    lam = 1e-4
    stopping_conditions = [1e-1, 5]
    
    flag = False
    error = get_error(solution, n_cameras, n_points_tot)
    print(f'Residual norm at start = {error}')
    # Keep stepping until stopping condition is met
    
    while not flag:
        solution, error, flag, lam = step(solution['is_calibrated'], solution, n_cameras, n_points_tot, lam, error, stopping_conditions)
    print(f'Residual norm at end = {error}')
    return solution

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', help="config file")
    args = parser.parse_args()

    problem = pickle.load(open(args.problem, 'rb'))
    solution = solve_ba_problem(problem)

    solution_path = args.problem.replace(".pickle", "-solution.pickle")
    pickle.dump(solution, open(solution_path, "wb"))
