#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MAXN = 1664512;
ll pr[MAXN],mo[MAXN],pm[MAXN];
bool y[MAXN];
map<ll,ll> trm;
// this is the mobius function prefix sum in O(n ^ (2/3)) using dirichlet convolution
ll mprf(ll x){
    if(x < MAXN) return pm[x];
    if(trm[x]) return trm[x];
    ll res = 1LL;
    // integer partitiioning
    for(ll i = 2,j; i <= x; i = j+1){
        ll prc = x / i;
        j = x / prc;
        res -= mprf(prc) * (j-i+1);
    }
    trm[x] = res;
    return res;
}
// this is the phi function prefix sum in O(n ^ (2/3)) using dirichlet convolution
ll phpr(ll x){
    ll res = 0LL;
    // integer partitiioning
    for(ll i = 1,j; i <= x; i = j+1){
        ll prc = x / i;
        j = (x / prc);
        res += (mprf(j) - mprf(i-1)) * prc * prc;
    }
    return (res+1)>>1;
}
int main(){
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    ll q; 
    cin >> q;
    // precompute the mobius function
    mo[1] = 1LL;
    ll at = 0;
    for(ll i = 2; i < MAXN; i++){
        if(!y[i]){
            at++;
            pr[at] = i, mo[i] = -1LL;
        }
        for(ll j=1;j<=at&&i*pr[j]<MAXN;j++){
            ll str= i * pr[j];
            y[str] = true;
            if(i % pr[j]){
                mo[str]= -mo[i];
            } else {
                mo[str] = 0LL; 
                break;
            }
        }
    }
    // precompute the mobius function prefix sum
    for(ll i = 1; i < MAXN; i++) pm[i] = pm[i-1] + mo[i];
    while(q--){
        // query the phi function and the mobius function prefix sum
        // example:
        // 3
        // 10
        // 100
        // 1000
        // output:
        // 17 10
        // 401 100
        // 178421 1000
        ll n;
        cin >> n;
        cout<< phpr(n) << " " << mprf(n) << "\n";
    }
}