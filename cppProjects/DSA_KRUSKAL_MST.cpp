#include <bits/stdc++.h>
using namespace std;
const int MAXN = 201;
const int INF = 1e5;
vector<int> parent(MAXN), ranke(MAXN);
void init (int n){
    parent.resize(n);
    ranke.resize(n);
    for(int i = 0; i < n; i++){
      parent[i] = i;
      ranke[i] = 1;
    }
    return;
  }
int find(int i){
  return ((parent[i] == i) ? i : (parent[i] = find(parent[i])));
}
void unite(int x,int y){
  int s1 = find(x),s2 = find(y);
  if(s1 != s2){
    if(ranke[s1] < ranke[s2]) parent[s1] = parent[s2];
    else if(ranke[s1] > ranke[s2]) parent[s2] = parent[s1];
    else parent[s2] = s1, ranke[s1]++;
  }
}
int main(){
  int a;
  cin >> a;
  vector<tuple<int,int,int>> edges;
  for(int i = 0; i < a; i++){
    for(int j = 0; j < a; j++){
      int a;
      cin >> a;
      if(a == 0){
        edges.push_back({0,i,j});        
      }
      else{
        edges.push_back({a,i,j});
      }
      //cout << a << " " << i << " " << j << "\n";
    }
  }      
  // sort the edges by weight
  sort(edges.begin(),edges.end());
  // initialize the union find data structure
  init (a);
  // find the minimum spanning tree
  // greedy algorithm
  int cost = 0, count = 0,rpec = 0;
  for(auto & e : edges){
    int x = get<0>(e);
    int y = get<1>(e);
    int w = get<2>(e);
    if(find(y) != find(w)){
      unite(y,w);
      cost+=x;
      if(x == 0) rpec++;
      if(++count == a-1) break;
    }
  }
  cout << count-rpec << "\n" << cost;
  return 0;
}