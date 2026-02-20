#include <bits/stdc++.h>
using namespace std;
class vertex {
public:
	int data;
    int store;
	vertex *left, *right;
	vertex(int x, char k) {
		data = x;
        store = k;
		left = nullptr;
		right = nullptr;
	}
};
class comp{
public:
    bool operator() (vertex*a, vertex*b){
        return a->data > b->data;
    }
};
// Ascii Encryption/Decryption
string ascii(string k){
  string res = "";
  for(auto c : k){
    int stuff = (int)c;
    for(int i = 0; i < 7; i++){
      if(stuff & 1){
        res.push_back('1');
      }
      else {
        res.push_back('0');
      }
      stuff>>=1;
    }
  }
  return res;
}
string unascii(string k){
  int at = 0;
  string ans = "";
  reverse(k.begin(),k.end());
  for(int i= 0; i < k.length(); i++){
    at *= 2;
    if(k[i] == '1')at++;
    if((i+1)%7==0){
      //cout << at << "\n";
      ans+=((char)at);
      //cout <<(char) at << "\n";
      at = 0;
    }
  }  
  reverse(ans.begin(),ans.end());
  return ans;
}
// Greedy Construction
void dfs(vertex* root, map<int,string> & encode, string curr){
    if(root == nullptr) return;
    if(root->store){
        encode[root->store] = curr;
        return;
    }
    dfs(root->left, encode, curr+'0');
    dfs(root->right, encode, curr+'1');
}
vertex* rooted;
string greedy(string k){
  map<char,int> frequency;
    for(auto c : k) frequency[c]++;
    priority_queue<vertex*,vector<vertex*>, comp> pq;
    for(auto c : frequency) {
        vertex* temp = new vertex(c.second, c.first);
        pq.push(temp);
    }
    while(pq.size() >= 2){
        auto l = pq.top();
        pq.pop();
        auto r = pq.top();
        pq.pop();
        auto res = new vertex(l->data + r->data, 0);
        res -> left = l;
        res -> right = r;
        pq.push(res);
    }
    auto root = pq.top();
    rooted = root;
    map<int,string> encode;
    dfs(root, encode, "");
    string resu = "";
    for(auto c : k) resu += encode[c];
    // for(auto c : k) cout << c << " " << encode[c] << "\n";
  return resu;
}
string ungreedy(string encode){
    vertex* at = rooted;
    string result;
    for(auto c : encode){
        if(at->store != 0) {
            result += at->store;
            at = rooted;
        }
        if(c=='0') at = at -> left;
        else at = at -> right;
    }
    result += at->store;
    return result;
}
void RunAscii(string k){
    string result = ascii(k);
    cout << "ASCII bits : " << result.length() << " " << result << "\n" << unascii(result) << "\n";
}
void RunHuffman(string k){
    string result = greedy(k);
    cout << "HUFFMAN bits : " << result.length() << " " << result << "\n" << ungreedy(result) << "\n";
}
int main() {
    string k;
    getline(cin,k);
    RunAscii(k);
    RunHuffman(k);
  //transform into bits?
  return 0;
}
