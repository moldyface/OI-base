#include <bits/stdc++.h>
using namespace std;
// KMP
// some string algorithms.
vector<int> getlps(string a){
  vector<int> res(a.length(),0);
  int len = 0;
  int i = 1;
  while(i < a.length()){
    if(a[i] == a[len]){
      len++;
      res[i] = len;
      i++;
    }
    else{
      if(len != 0) len = res[len-1];  
      else{
        res[i] = 0;
        i++;
      }
    }
  }
  return res;
}
vector<int> KMP(string a, string b){
  vector<int> ans;
  vector<int> pi = getlps(a);
  int i = 0, j = 0;
  while(i < b.length()){
    if(a[j] == b[i]){
      i++;
      j++;
    }
    if(j == a.length()){
      ans.push_back(i-j);
      j = pi[j-1];
    }
    else if(i < b.length() && a[j] != b[i]){
      if(j != 0){
        j = pi[j-1];
      }
      else i++;
    }
  }
  return ans;
}
bool KMP_exist(string a, string b){
  vector<int> pi = getlps(a);
  int i = 0, j = 0;
  while(i < b.length()){
    if(a[j] == b[i]){
      i++;
      j++;
    }
    if(j == a.length()){
      return true;
    }
    else if(i < b.length() && a[j] != b[i]){
      if(j != 0){
        j = pi[j-1];
      }
      else i++;
    }
  }
  return false;
}
int non_overlap(string a, string b){
  // KMP with a 
  vector<int> dp(b.length()+2,0);
  vector<int> pi = getlps(a);
  int i = 0, j = 0;
  while(i < b.length()){
    dp[i+1] = max(dp[i+1],dp[i]);
    if(a[j] == b[i]){
      i++;
      j++;
    }
    if(j == a.length()){
      dp[i+1] = max(dp[i+1],dp[i-j+1]+1);
      j = pi[j-1];
    }
    else if(i < b.length() && a[j] != b[i]){
      if(j != 0){
        j = pi[j-1];
      }
      else i++;
    }
  }
  return max(dp[b.length()],dp[b.length()+1]);
}

vector<int> zarray(string s){
  int n = s.length();
  vector<int> z(n,0);
  int x=0,y=0;
  for(int i = 0; i < n; i++){
    z[i] = max(0,min(z[i-x],y-i+1));
    while((i+z[i]) < n && s[z[i]] == s[i+z[i]]){
      x = i, y = i+z[i]; z[i]++;
    }
  }
  return z;
}
string shortestperiod(string a){
  //i made this myself :)
  vector<char> ans = {a[0]};
  vector<char> temp = {};
  int at = 0;
  int rect = 0;
  for(int i = 1; i < a.length(); i++){
    temp.push_back(a[i]);
    //cout << ans << " " << temp << endl;
    if(ans[(at % ans.size())] != temp[at+rect]){
      ans.push_back(temp[rect]);
      rect++;
      at = 0;
    }
    else{
      at++;
    }
  }
  //now check
  if(a.length() % ans.size() == 0){
    string res = "";
    for(auto c : ans) res += c;
    return res;
  }
  return a;
}
int main() {
//   string a;
//   cin >> a;
//   cout << shortestperiod(a);
    cout << "Pattern-string\n";
    string a,b;
    cin >> a >> b;
    vector<int> answer = KMP(a,b);
    cout << b << "\n";
    for(auto c : answer){
        for(int i = 0; i < c; i++) cout << " ";
        cout << a << "\n";
    }
}