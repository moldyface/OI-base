#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF = 1e14;
// Bellman Ford : O(mn)
// get shortest path, negative edges.support
int main() {
  ll a,b;
  cin >> a >> b;
  vector<array<ll,3>> edges;
  vector<ll> dist(a,INF);
  dist[0] = 0;
  for(int i = 0; i < b; i++){
    ll c,d,e;
    cin >> c >> d >> e;
    c--;d--;
    edges.push_back({c,d,e});
  }
  for(int i = 0; i < a; i++){
    for(auto [x,y,z] : edges){
      dist[y] = min(dist[y],dist[x] + z);
    }
  }
  for(auto [x,y,z] : edges){
    if(dist[y] > (dist[x] + z)){
      cout << "Negative Cycle detected.";
      return 0;
    }
  }
  cout << "No Negative Cycle.\n";
  for(int i = 0; i < a; i++){
    cout << (dist[i] > 1e9 ? 1000000000 : dist[i])<< " ";
  }
  return 0;
}
