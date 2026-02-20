#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef __int128 i1;
typedef vector<vector<ll>> matrix;
mt19937_64 mt1(time(nullptr));

matrix randommatrix(ll a){
    matrix k(a, vector<ll>(a));
    for(ll i = 0; i < a; ++i)
        for(ll j = 0; j < a; ++j)
            k[i][j] = mt1() % 100;
    return k;
}

ll nextPowerOfTwo(ll n){
    if(n <= 1) return 1;
    ll power = 1;
    while(power < n) power <<= 1;
    return power;
}

matrix resizeMatrix(const matrix &mat, ll newrow, ll newcolumns){
    matrix res(newrow, vector<ll>(newcolumns, 0));
    ll rows = min((ll)mat.size(), newrow);
    ll cols = min((ll)mat[0].size(), newcolumns);
    for(ll i = 0; i < rows; ++i)
        for(ll j = 0; j < cols; ++j)
            res[i][j] = mat[i][j];
    return res;
}

inline void add_inplace(matrix &res, const matrix &a, const matrix &b, ll siz, ll sign = 1){
    for(ll i = 0; i < siz; ++i)
        for(ll j = 0; j < siz; ++j)
            res[i][j] = a[i][j] + (sign == 1 ? b[i][j] : -b[i][j]);
}

inline void sub_inplace(matrix &res, const matrix &a, const matrix &b, ll siz){
    for(ll i = 0; i < siz; ++i)
        for(ll j = 0; j < siz; ++j)
            res[i][j] = a[i][j] - b[i][j];
}

matrix naive_mult(const matrix &a, const matrix &b){
    ll n = a.size();
    matrix res(n, vector<ll>(n, 0));
    for(ll i = 0; i < n; i++) {
        for(ll k = 0; k < n; k++) {
            ll aik = a[i][k];
            for(ll j = 0; j < n; j++) {
                res[i][j] += aik * b[k][j];
            }
        }
    }
    return res;
}

void strassen_mult(matrix &res, const matrix &a, const matrix &b, ll n) {
    if(n <= 64) {
        auto naive = naive_mult(a, b);
        for(ll i = 0; i < n; ++i)
            for(ll j = 0; j < n; ++j)
                res[i][j] = naive[i][j];
        return;
    }
    
    ll ns = n >> 1;
    matrix a11(ns, vector<ll>(ns)), a12(ns, vector<ll>(ns)),
           a21(ns, vector<ll>(ns)), a22(ns, vector<ll>(ns)),
           b11(ns, vector<ll>(ns)), b12(ns, vector<ll>(ns)),
           b21(ns, vector<ll>(ns)), b22(ns, vector<ll>(ns));
    matrix temp1(ns, vector<ll>(ns)), temp2(ns, vector<ll>(ns)),
           m1(ns, vector<ll>(ns)), m2(ns, vector<ll>(ns)),
           m3(ns, vector<ll>(ns)), m4(ns, vector<ll>(ns)),
           m5(ns, vector<ll>(ns)), m6(ns, vector<ll>(ns)),
           m7(ns, vector<ll>(ns));
    matrix s1(ns, vector<ll>(ns)), s2(ns, vector<ll>(ns)),
           s3(ns, vector<ll>(ns)), s4(ns, vector<ll>(ns)),
           s5(ns, vector<ll>(ns)), s6(ns, vector<ll>(ns)),
           s7(ns, vector<ll>(ns)), s8(ns, vector<ll>(ns));
    
    #pragma unroll(4)
    for(ll i = 0; i < ns; ++i){
        #pragma unroll(4)
        for(ll j = 0; j < ns; ++j) {
            a11[i][j] = a[i][j];
            a12[i][j] = a[i][j + ns];
            a21[i][j] = a[i + ns][j];
            a22[i][j] = a[i + ns][j + ns];
            b11[i][j] = b[i][j];
            b12[i][j] = b[i][j + ns];
            b21[i][j] = b[i + ns][j];
            b22[i][j] = b[i + ns][j + ns];
        }
    }
    
    // Precompute all needed sums once
    add_inplace(s1, a11, a22, ns);      // s1 = a11 + a22
    add_inplace(s2, b11, b22, ns);      // s2 = b11 + b22
    add_inplace(s3, a21, a22, ns);      // s3 = a21 + a22
    sub_inplace(s4, b12, b22, ns);      // s4 = b12 - b22
    sub_inplace(s5, b21, b11, ns);      // s5 = b21 - b11
    add_inplace(s6, a11, a12, ns);      // s6 = a11 + a12
    sub_inplace(s7, a21, a11, ns);      // s7 = a21 - a11
    add_inplace(s8, b11, b12, ns);      // s8 = b11 + b12
    sub_inplace(temp1, a12, a22, ns);   // temp1 = a12 - a22
    add_inplace(temp2, b21, b22, ns);   // temp2 = b21 + b22
    
    // Compute the 7 products
    strassen_mult(m1, s1, s2, ns);      // m1 = (a11+a22)*(b11+b22)
    strassen_mult(m2, s3, b11, ns);     // m2 = (a21+a22)*b11
    strassen_mult(m3, a11, s4, ns);     // m3 = a11*(b12-b22)
    strassen_mult(m4, a22, s5, ns);     // m4 = a22*(b21-b11)
    strassen_mult(m5, s6, b22, ns);     // m5 = (a11+a12)*b22
    strassen_mult(m6, s7, s8, ns);      // m6 = (a21-a11)*(b11+b12)
    strassen_mult(m7, temp1, temp2, ns);// m7 = (a12-a22)*(b21+b22)
    
    // Combine results with fewer intermediate operations
    #pragma unroll(4)
    for(ll i = 0; i < ns; ++i){
        #pragma unroll(4)
        for(ll j = 0; j < ns; ++j) {
            // c11 = m1 + m4 - m5 + m7
            res[i][j] = m1[i][j] + m4[i][j];
            res[i][j] += m7[i][j] - m5[i][j];
            
            // c12 = m3 + m5
            res[i][j + ns] = m3[i][j] + m5[i][j];
            
            // c21 = m2 + m4
            res[i + ns][j] = m2[i][j] + m4[i][j];
            
            // c22 = m1 - m2 + m3 + m6
            res[i + ns][j + ns] = m1[i][j] - m2[i][j];
            res[i + ns][j + ns] += m3[i][j] + m6[i][j];
        }
    }
}

matrix multiply(const matrix &a, const matrix &b){
    
    ll n = nextPowerOfTwo(max({a.size(), a[0].size(), b[0].size()}));
    matrix ap = resizeMatrix(a, n, n);
    matrix bp = resizeMatrix(b, n, n);
    matrix pres(n, vector<ll>(n, 0));
    
    strassen_mult(pres, ap, bp, n);
    
    matrix res(a.size(), vector<ll>(b[0].size(), 0));
    for(ll i = 0; i < a.size(); ++i)
        for(ll j = 0; j < b[0].size(); ++j)
            res[i][j] = pres[i][j];
    return res;
}

void outputmatrix(const matrix &a){
    for(const auto &c : a){
        for(auto b : c) cout << b << " ";
        cout << "\n";
    }
}

int main() {
    cout << "Code Begins now\n";
    matrix A = randommatrix(1024);
    matrix B = randommatrix(1024);
    cout << "Starting time: " << clock() << "\n";
    auto starttime = clock();
    matrix C = naive_mult(A, B);
    cout << "Time took: `" << clock() - starttime   << "\n";
    cout << "YES!";
    return 0;
}