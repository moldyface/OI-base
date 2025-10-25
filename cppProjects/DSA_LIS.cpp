#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
inline void findlis(){
  ll a;
  cin >> a;
  vector<ll> nums(a);
  for(auto & c : nums){
    cin >> c;
  }
  vector<ll> dp(a);
  vector<ll> tops = {};
  for(ll i = 0; i < a; i++){
    auto place = lower_bound(tops.begin(),tops.end(),nums[i]);
    dp[i] = place-tops.begin()+1;
    if(place == tops.end()) tops.push_back(nums[i]);
    else *place = nums[i];
  }
  int maxlen = tops.size();
  vector<int> antwort;
  for(int i = a-1; i >= 0; i--){
    if(dp[i] == maxlen){
      maxlen--;
      antwort.push_back(nums[i]);
    }
  }
  reverse(antwort.begin(),antwort.end());
  for(auto c : antwort){
    cout << c << " ";
  }
  return;
}
int main(){
  findlis();
}