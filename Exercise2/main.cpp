#include <iostream>
#include <fstream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

int main()
{
    double a1 = 5.547001962252291e-01;
    double *p_a1 = &a1;

    double a1_2 = -3.770900990025203e-02;
    double *p_a1_2 = &a1_2;
    double a2_2 = -5.540607316466765e-01;
    double *p_a2_2 = &a2_2;
    double a3_2 = -5.547001955851905e-01;
    double *p_a3_2 = &a3_2;

    double a3 =  8.320502943378437e-01;
    double *p_a3 = &a3;

    double a1_4 = -9.992887623566787e-01;
    double *p_a1_4 = &a1_4;
    double a2_4 = -8.324762492991313e-01;
    double *p_a2_4 = &a2_4;
    double a3_4 = -8.320502947645361e-01;
    double *p_a3_4 = &a3_4;

    MatrixXd A_1 = MatrixXd::Ones(2,2);
    A_1 << *p_a1, *p_a1_2,
        *p_a3,*p_a1_4;

    MatrixXd A_2 = MatrixXd::Ones(2,2);
    A_2 << *p_a1, *p_a2_2,
        *p_a3,*p_a2_4;

    MatrixXd A_3 = MatrixXd::Ones(2,2);
    A_3 << *p_a1, *p_a3_2,
        *p_a3,*p_a3_4;

    // ------------------

    double b1_1 = -5.169911863249772e-01;
    double *p_b1_1 = &b1_1;
    double b1_2 = 1.672384680188350e-01;
    double *p_b1_2 = &b1_2;

    double b2_1 = -6.394645785530173e-04;
    double *p_b2_1 = &b2_1;
    double b2_2 =  4.259549612877223e-04;
    double *p_b2_2 = &b2_2;

    double b3_1 = -6.400391328043042e-10;
    double *p_b3_1 = &b3_1;
    double b3_2 = 4.266924591433963e-10;
    double *p_b3_2 = &b3_2;

    VectorXd b_1 = VectorXd::Ones(2,1);
    b_1 << *p_b1_1,
        *p_b1_2;

    VectorXd b_2 = VectorXd::Ones(2,1);
    b_2 << *p_b2_1,
        *p_b2_2;

    VectorXd b_3 = VectorXd::Ones(2,1);
    b_3 << *p_b3_1,
        *p_b3_2;

    // ------------------

    VectorXd x = VectorXd::Ones(2,1);
    x << -1.0e+0,
        -1.0e+00;
    VectorXd *p_x = &x;

    Eigen::FullPivLU<Matrix2d> lu_1(A_1);
    VectorXd x_lu_1 = A_1.fullPivLu().solve(b_1);
    VectorXd *p_lu_1 = &x_lu_1;
    Eigen::FullPivLU<Matrix2d> lu_2(A_2);
    VectorXd x_lu_2 = A_2.fullPivLu().solve(b_2);
    VectorXd *p_lu_2 = &x_lu_2;
    Eigen::FullPivLU<Matrix2d> lu_3(A_3);
    VectorXd x_lu_3 = A_3.fullPivLu().solve(b_3);
    VectorXd *p_lu_3 = &x_lu_3;

    Eigen::HouseholderQR< Matrix2d > qr_1(A_1);
    VectorXd x_qr_1 = A_1.householderQr().solve(b_1);
    VectorXd *p_qr_1 = &x_qr_1;
    Eigen::HouseholderQR< Matrix2d > qr_2(A_2);
    VectorXd x_qr_2 = A_2.householderQr().solve(b_2);
    VectorXd *p_qr_2 = &x_qr_2;
    Eigen::HouseholderQR< Matrix2d > qr_3(A_3);
    VectorXd x_qr_3 = A_3.householderQr().solve(b_3);
    VectorXd *p_qr_3 = &x_qr_3;

    std::ofstream outFile("Rel_Errors.txt");
    double err_qr_1;
    err_qr_1 = ((*p_qr_1 - *p_x).norm())/((*p_x).norm());
    outFile << "QR decomposition relative error 1: " << err_qr_1 << endl;
    double err_qr_2;
    err_qr_2 = ((*p_qr_2 - *p_x).norm())/((*p_x).norm());
    outFile << "QR decomposition relative error 2: " << err_qr_2 << endl;
    double err_qr_3;
    err_qr_3 = ((*p_qr_3 - *p_x).norm())/((*p_x).norm());
    outFile << "QR decomposition relative error 3: " << err_qr_3 << endl;

    outFile << "\n" << endl;

    double err_lu_1;
    err_lu_1 = ((*p_lu_1 - *p_x).norm())/((*p_x).norm());
    outFile << "LU matrix factorization relative error 1: " << err_lu_1 << endl;
    double err_lu_2;
    err_lu_2 = ((*p_lu_2 - *p_x).norm())/((*p_x).norm());
    outFile << "LU matrix factorization relative error 2: " << err_lu_2 << endl;
    double err_lu_3;
    err_lu_3 = ((*p_lu_3 - *p_x).norm())/((*p_x).norm());
    outFile << "LU matrix factorization relative error 3: " << err_lu_3 << endl;

    outFile.close();
    return 0;
}
