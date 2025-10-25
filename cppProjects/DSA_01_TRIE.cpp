#include <bits/stdc++.h>
#pragma optimize("Ofast")
// MAXIMUM xor.
using namespace std;
typedef long long ll;
const ll MAXN = 36;
struct vertex{
  int next[2];
  vertex(){
    next[0]=next[1] = -1;
  }
};
vector<vertex> trie(1);
void add_num(ll a){
  vector<int> res(MAXN);
  for(int i = 0; i < MAXN; i++){
    res[i] = a & 1;
    a >>= 1;
  }
  reverse(res.begin(),res.end());
  ll v = 0;
  for(int i = 0; i < MAXN; i++){
    if(trie[v].next[res[i]] == -1){
      trie[v].next[res[i]] = trie.size();
      trie.emplace_back();
    }
    v = trie[v].next[res[i]];
  }
}
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0); cout.tie(0);
  ll a;
  cin >> a;
  vector<ll> m(a);
  for(auto & c : m) cin >> c;
  for(auto c : m) add_num(c);
  ll ans = 0;
  for(int i = 0; i < a; i++){
    ll v = 0;
    ll curr = 0;
    ll cval = 1LL << (MAXN - 1);
    vector<int> res(MAXN);
    ll st = m[i];
    for(int i = 0; i < MAXN; i++){
      res[i] = st & 1;
      st >>= 1;
    }
    reverse(res.begin(),res.end());
    
    for(int j = 0; j < MAXN; j++){
      if(trie[v].next[1-res[j]] != -1) {
        v = trie[v].next[1-res[j]];
        curr += cval;
      } else {
        v = trie[v].next[res[j]];
      }
      cval >>= 1;
    }
    ans = max(ans,curr);
  }
  cout << ans;
  return 0;
}
