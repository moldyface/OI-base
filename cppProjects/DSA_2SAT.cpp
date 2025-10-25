#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
ll n;
vector<vector<ll>> g, ig; 
vector<ll> comp, topo;
vector<char> ans; 
vector<bool> vis;
inline void init(ll happ) {
  n = happ;
  g.assign(2*n, vector<ll>());
  ig.assign(2*n, vector<ll>());
  comp.resize(2 * n);
  vis.resize(2 *n);
  ans.resize(n);
}
inline void addedge(ll u, ll v) {
    g[u].push_back(v);
    ig[v].push_back(u);
}
inline void add(ll i, bool f, ll j, bool g) {
    addedge(i + (f ? n : 0), j + (g ? 0 : n));
    addedge(j + (g ? n : 0), i + (f ? 0 : n));
}

//tofu sort
inline void dfs(ll u) {
  vis[u] = true;
  for (const auto &v : g[u])
    if (!vis[v]) dfs(v);
  topo.push_back(u);
}

    // Extscc
inline void scck(ll u, ll id) {
  vis[u] = true;
  comp[u] = id;
  for (const auto &v : ig[u])
    if (!vis[v]) scck(v, id);
  }
   inline bool solve() {
      //initing
        fill(vis.begin(), vis.end(), false);

        for (ll i = 0; i < 2 * n; i++)
            if (!vis[i]) dfs(i);
      //toposorting
      //initing for kosa
        fill(vis.begin(), vis.end(), false);
        reverse(topo.begin(), topo.end());
      
        ll id = 0;
        for (const auto &v : topo)
            if (!vis[v]) scck(v, id++);
      
        for (ll i = 0; i < n; i++) {
            if (comp[i] == comp[i + n]) {cout << "IMPOSSIBLE" ; return false;}
            ans[i] = (comp[i] > comp[i + n] ? '+' : '-');
        }
       cout << "POSSIBLE\n";
      for(auto c : ans){
        cout << (c == '+' ? 1 : 0) << " ";
      }
        return true;
    }

signed main(){
  ll d,i;
  cin >> d >> i;
  init(d);
  for(ll j = 0; j < i; j++){
    ll tr,a,k,s;
    cin >> k >> tr >> s >> a;
    k--;
    s--;
    add(k,tr,s,a);
  }
  solve();
  return 0;
}
