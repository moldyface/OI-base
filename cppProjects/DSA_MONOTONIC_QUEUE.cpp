#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
// A311
// what does this do?
// this is a dp problem that finds the maximum value of a subarray sum, given that the subarray must not "jump" more than m elements.
// example:
// 5 2
// 1 2 3 4 5
// output:
// 15
int main() {
  ll n,m;
  cin >> n >> m;
  vector<ll> num(n);
  vector<ll>  dp(n);
  deque<ll>  monodq;
  for(auto & c : num) cin >> c;
  // dp[i] : max sum, choosing i.
  //transition:
  // dp[i] = max(dp[j]) , j in bounds, A[i]
  monodq.push_back(0);
  dp[0] = num[0];
  // initial population
  for(int i = 1; i < n; i++){
    while(!monodq.empty() && monodq.front() < (i - m))  
      monodq.pop_front();
    // remove expired elements
    if(!monodq.empty())
      dp[i] = dp[monodq.front()] + num[i];
    // get_max

    while(!monodq.empty() && dp[monodq.back()] < dp[i])  monodq.pop_back();
    monodq.push_back(i);
  }
  cout << dp[n - 1];
  return 0;
}
