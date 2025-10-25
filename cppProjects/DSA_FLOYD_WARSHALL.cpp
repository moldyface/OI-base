#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int INF = 1e9;
const int MAXN = 405;
int dist[MAXN][MAXN];
vector<int> dis(MAXN,INF);
void floydwarshall(){
  int n,m;
  cin >> n >> m;
  // floyd Warshall to find min dist between all paths.
  // O(n^3)
  for(int i = 0; i < MAXN; i++){
    for(int j = 0; j < MAXN; j++){
      dist[i][j] = INF;
    }
  }
  for(int i = 0; i < m; i++){
    int a,b,c;
    cin >> a >> b >> c;
    dist[a][b] = c;
    dist[b][a] = c;
  }
  for(int i = 0; i < MAXN; ++i){
    for(int j = 0; j < MAXN; ++j){
      for(int k = 0; k < MAXN; ++k){
        dist[j][k] = min(dist[j][k],dist[j][i]+dist[i][k]);
      }
    }
  }
} 
int main(){
  floydwarshall();
}