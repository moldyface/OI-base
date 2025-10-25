#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MAXN = 1e5;
vector<bool> rer(MAXN+1,1);
vector<ll> prim;
vector<ll> nu(MAXN+1,0);
// that calculates how much things from 1 - a, 1 - b gcd = c.
ll val(ll a,ll b,ll c){
    if(b>a)swap(a,b);
    ll res = 0;
    ll bnd = a/c;
    for(ll d = 1; d <= bnd; d++){
        res += nu[d] * (a / (d*c)) * (b / (d*c));
    }
    return res;
}
inline void mobius(){
    ll to = 1001;
    for(ll p = 2; p <= to; p++){
        if(rer[p]){
            for(ll c = p<<1; c <= MAXN; c += p)
                rer[c] = 0;
        }
    }
    for(ll p = 2; p <= MAXN; p++){
        if(rer[p]) prim.push_back(p);
    }
    nu[1] = 1;
    for(ll i = 2; i <= MAXN;i++){
        ll st = i;
        ll res = 1;
        bool flag = false;
        if(rer[st]){
            nu[st] = -1;
            continue;
        }
        for(auto c : prim){
            if(st % c == 0){
                st /= c;
                if(st % c == 0){
                    flag = true;
                }
                res = -nu[st];
                break;
            }
            if(c > st) break;
        }
        if(flag){
             nu[i] = 0;
        } 
        else {
            nu[i]  = res;
        }
    }
}
int main(){
    mobius();
}