#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MAXN = 1e5;
const ll INF = 1e8;
vector<pair<ll,ll>> graph[MAXN];
ll dist[MAXN]= {INF};
int main() {
    int n,vr;
    cout << "number of edges, vertices:\n";
    cin >> n >> vr;
    for(int i = 0; i < n; ++i){
        int a,b,c;
        cin >> a >> b >> c;
        graph[a].push_back({b,c});
    }
    int src;
    cout << "source\n";
    cin >> src;
    for(int i = 1; i <= vr; i++) dist[i] = INF;
    dist[src] = 0;
    priority_queue<pair<ll,ll>, vector<pair<ll,ll>>, greater<pair<ll,ll>>> pq; // Min-heap
    pq.push({0, src});
    while (!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();
        if (d > dist[u]) continue;
        for (auto edge : graph[u]) {
            int v = edge.first;
            int w = edge.second;

            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    for(int i = 1; i <= vr; i++){
        cout << i << " : " << dist[i] << "\n";
    }
    return 0;
}