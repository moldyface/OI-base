#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
// String Trie.
const int K = 26;
struct Vertex {
    int next[K];
    bool output = false;

    Vertex() {
        fill(begin(next), end(next), -1);
    }
};
vector<Vertex> trie(1);
void add_string(string s) {
    int v = 0;
    for (char ch : s) {
        int c = ch - 'a';
        if (trie[v].next[c] == -1) {
            trie[v].next[c] = trie.size();
            trie.emplace_back();
        }
        v = trie[v].next[c];
    }
    trie[v].output = true;
}
bool find_string(string s){
    ll v = 0;
    for(auto c : s){
        if(trie[v].next[c - 'a'] != -1){
            v = trie[v].next[c-'a'];
        }
        else return false;
    }
    return true;
}

string find_prefix(string s){
    string res = "";
    ll v = 0;
    for(auto c : s){
        if(trie[v].next[c - 'a'] != -1){
            v = trie[v].next[c-'a'];
            res += c;
        }
        else break;
    }
    return res;
}
int main(){
    int a;
    cout << "strings to add:\n";
    cin >> a;
    cout << "strings:\n";
    for(int i = 0; i < a; i++){
        string b; cin >> b; add_string(b);
    }
    int b;
    cout << "strings to find:\n";
    cin >> b;
    cout << "strings\n";
    for(int i = 0; i < b; i++){
        string c;
        cin >> c;
        cout << "Longest prefix : " << find_prefix(c) << "\n";
    }
}