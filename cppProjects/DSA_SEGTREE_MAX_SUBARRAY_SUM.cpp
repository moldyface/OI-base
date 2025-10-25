#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MAXN = 1e6+5;
vector<ll> a(MAXN);

// ARRAY to be turned into segment tree
// Segment tree that finds: Max subarray sum with point updates.
// Example:
/*
5 3
1 2 -3 5 -1
2 6
3 1
2 -2
_
cout << 9,13,6;
*/
// Segment Tree && Lazy Propogation
// Lazy propogation is O(log(n))
// segment tree is O(log n)
ll c;
ll tree[MAXN], tmax[MAXN], lmax[MAXN], rmax[MAXN];
inline ll query(ll id, ll L, ll R, ll QL, ll QR){ // range [L, R], query range [QL, QR]
  if (QR < L || R < QL) // no intersection between [L, R] & [QL, QR]
    return 0;
  if (QL <= L && R <= QR) // [L, R] is fully inside [QL, QR]
    return tree[id];
  ll mid = L + R;
  mid >>=1;
  return (query(id * 2, L, mid, QL, QR)+query(id * 2 + 1, mid + 1, R, QL, QR));
}

inline void build(ll id, ll l, ll r){
  if(l == r){
    lmax[id] = rmax[id] = tmax[id] = tree[id] = a[l];
    return;
  }
  
  ll mid = l + r;
  mid >>=1;
  build(id<<1, l , mid);
  build(id<<1|1 , mid + 1 , r);
  tree[id] = tree[id<<1] + tree[id<<1|1];
  lmax[id] = max(lmax[id<<1] , query(1,1,c,l,mid) + lmax[id<<1|1]);
  rmax[id] = max(rmax[id<<1|1] , query(1,1,c,mid+1,r) + rmax[id<<1]);
  tmax[id] = max(lmax[id],rmax[id]);
  tmax[id] = max(tmax[id],tmax[id<<1]);
  tmax[id] = max(tmax[id],tmax[id<<1|1]);
  tmax[id] = max(tmax[id],rmax[id<<1] + lmax[id<<1|1]);
}


inline void update_val(ll id, ll L, ll R, ll x, ll val){
  if (L == R)
  {
    lmax[id] = rmax[id] = tmax[id] = tree[id] = val;
    return;
  }
  ll mid = L + R;
  mid >>=1;
  if (x <= mid)
    update_val(id <<1, L, mid, x, val);
  else
    update_val(id<<1|1, mid + 1, R, x, val);
  
  tree[id] = tree[id<<1] + tree[id<<1|1];
  lmax[id] = max(lmax[id<<1] , query(1,1,c,L,mid) + lmax[id<<1|1]);
  rmax[id] = max(rmax[id<<1|1] , query(1,1,c,mid+1,R) + rmax[id<<1]);
  tmax[id] = max(lmax[id],rmax[id]);
  tmax[id] = max(tmax[id],tmax[id<<1]);
  tmax[id] = max(tmax[id],tmax[id<<1|1]);
  tmax[id] = max(tmax[id],rmax[id<<1] + lmax[id<<1|1]);
}



inline void tree_print(int n)
{
  int i, j, h = ceil(log2(n));
  for(i=0 ; i<=h ; ++i)
  {
    for(j=0 ; j<pow(2, i) ; ++j)
      cout<<tree[int(pow(2, i)-1 + j)]<<' ';
    cout<<endl;
  }
}

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  ll s;
  cin >> c >> s;
  a.resize(c+1);
  for(ll i = 1; i <= c; i++) cin >> a[i];
  build(1,1,c);
  while(s--){
    ll b,d;
    cin >> b >> d;
    update_val(1,1,c,b,d);
    cout << max(0LL,tmax[1]) << "\n";
  }
  return 0;
}
