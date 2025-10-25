#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MAXN = 1e5;
const ll INF = 1e9;
vector<ll> G[MAXN];
// DSA Tarjan to find strongly connected components.
vector<bool> in_stack(MAXN);
vector<ll> st(MAXN),low(MAXN),scc(MAXN),s(MAXN);
stack<ll> sting;
ll tin = 0, k=0;

void dfs(ll u){
  st[u] = low[u] = ++tin;
  s.emplace_back(u);
  in_stack[u]=true;
  for(ll v : G[u]){
    if(!st[v]){
      dfs(v);
      low[u] = min(low[u],low[v]);
    }
    else if(in_stack[v]){
      low[u] = min(low[u],st[v]);
    }
  }
  if(st[u] == low[u]){
    ++k;
    for(ll x = -1; x != u; s.pop_back()){
      x = s.back();
      scc[x] = k;
      in_stack[x] = false;
    }
  }
}
int main() {
  ll a,b;
  cin >> a >> b; 
  for(ll i = 0; i < b; i ++){
    ll c,d;
    cin >> c >> d;
    c--;
    d--;
    G[c].push_back(d);
    }
    cout << "Strongly connected components are:\n";
  for (ll i = 0; i < a; ++i)
      if (!st[i])
          dfs(i);
  // strongly connected components;
  for(ll i = 1; i <= a; i++){
    cout << i << " ";
  }
  cout << "\n";
  for(ll i = 0; i < a; i++){
    cout << scc[i] << " ";
  }
  return 0;
}

