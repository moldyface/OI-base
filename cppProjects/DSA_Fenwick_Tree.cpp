#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MAXN = 1e5+4;
ll n,m, c[MAXN], r[MAXN];


// Returns the lowest bit
// 2 returns 2
// and similarily
ll lowbit(ll x){ 
  return (x&(-x)); 
}


//Query the value at pos x.
ll query(ll x){
  ll res = 0;
  for(; x > 0; x -= lowbit(x)) res +=  c[x];
  return res;
}


//modify ALL values from 0-x + r.
void modify(ll x, ll r){
  for(; x <= n; x += lowbit(x)) c[x] += r;
}


//modify ALL values from l - r;;
void partialmodify(ll l, ll r, ll change){
  modify(l,change);
  modify(r+1,-change);
}
int main(){
  cin >> n >> m;
  for(int i = 1; i <= n; i++){
    cin >> r[i];
    modify(i,r[i]-r[i-1]);
  }
  while(m--){
    ll que;
    cin >> que;
    if(que&1){
      ll l,o,n;
      cin >> l >> o >> n;
      partialmodify(l,o,n);
    }
    else{
      ll ry;
      cin >> ry;
      cout << query(ry) << "\n"; 
    }
  }
  return 0;
}