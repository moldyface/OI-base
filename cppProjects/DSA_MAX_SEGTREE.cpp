#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MAXN = 4e5+5;

vector<ll> a(MAXN);

// Segment Tree && Lazy Propagation
ll tree[MAXN], lazy[MAXN];

const ll INF = LLONG_MIN; // For query no-intersection
// segment tree that finds the max value in a range with lazy propagation
// this is 1-indexed
void build(ll id, ll l, ll r){
  if(l == r){
    tree[id] = a[l];
    return;
  }
  
  ll mid = l + (r - l) / 2;
  build(id * 2, l, mid);
  build(id * 2 + 1, mid + 1, r);
  tree[id] = max(tree[id*2], tree[id*2+1]);
}

void push_down(ll id, ll L, ll mid, ll R){
  tree[id*2] += lazy[id];
  tree[id*2+1] += lazy[id];
  lazy[id*2] += lazy[id];
  lazy[id*2+1] += lazy[id];
  lazy[id] = 0;
}

ll query(ll id, ll L, ll R, ll QL, ll QR){
  if (QR < L || R < QL)
    return INF;
  if (QL <= L && R <= QR)
    return tree[id];
  ll mid = L + (R - L) / 2;
  push_down(id, L, mid, R);
  return max(query(id*2, L, mid, QL, QR), 
            query(id*2+1, mid+1, R, QL, QR));
}

void update_val(ll id, ll L, ll R, ll x, ll val){
  if (L == R){
    tree[id] += val;
    return;
  }
  ll mid = L + (R - L) / 2;
  if (x <= mid)
    update_val(id*2, L, mid, x, val);
  else
    update_val(id*2+1, mid+1, R, x, val);
  tree[id] = max(tree[id*2], tree[id*2+1]);
}

void update_range(ll id, ll L, ll R, ll QL, ll QR, ll val){
  if (QR < L || R < QL)
    return;
  if (QL <= L && R <= QR){
    tree[id] += val; // Max increases by val
    lazy[id] += val;
    return;
  }
  ll mid = L + (R - L) / 2;
  push_down(id, L, mid, R);
  update_range(id*2, L, mid, QL, QR, val);
  update_range(id*2+1, mid+1, R, QL, QR, val);
  tree[id] = max(tree[id*2], tree[id*2+1]);
}

void tree_print(int n){
  int h = ceil(log2(n));
  for(int i = 0; i <= h; i++){
    int start = (1 << i);
    int end = (1 << (i+1)) - 1;
    for(int j = start; j <= end; j++){
      if (j >= MAXN) break;
      cout << tree[j] << ' ';
    }
    cout << endl;
  }
}

int main() {
  // input the number of cases, the number of segments, and the number of queries
  // then, for each query, input the start and end of the segment and the value to add
  // then, print the result of the query
  // example:
  // 3 10 3
  // 1 3 5
  // 2 4 3
  // cout << T,T;
  ll c, s, r;
  cin >> c >> s >> r;
  a.resize(c+1); // 1-based indexing
  build(1, 1, c);
  while(r--){
    ll e, b, d;
    cin >> e >> b >> d;
    b--;
    ll res = query(1, 1, c, e, b);
    if (s < res + d) cout << "N\n";
    else {
      cout << "T\n";
      update_range(1, 1, c, e, b, d);
    }
  //  update_range(1, 1, c, e, b, d);
   // tree_print(c);
  }
  return 0;
}