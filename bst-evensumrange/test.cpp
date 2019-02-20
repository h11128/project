#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<string>
#include<sstream>
#include <fstream>
#include <streambuf>
#include <vector>
using namespace std;

class BST
{
    struct node
    {   long long data;
        node* left;
        node* right;
        int evensum;
    };

    node* root;

    node* insert(long long x, node* t)
    {
        if(t == NULL)
        {   t = new node;
            t->data = x;
            t->evensum = -1;
            update_evensum(x, t);
            t->left = t->right = NULL;
        }
        else if(x <= t->data)
        {   update_evensum(x, t);
            t->left = insert(x, t->left);
        }
        else if(x > t->data)
        {   update_evensum(x, t);
            t->right = insert(x, t->right);
        }
        return t;
    }

    void inorder(node* t)
    {   if(t == NULL)
            return;
        inorder(t->left);
        cout << t->data << ":"<<t->evensum<<" ";
        inorder(t->right);
    }


public:
    BST()
    {       root = NULL;  }

    void insert(long long x)
    {        root = insert(x, root);    }

    int evenSumRange()
    {        return root->evensum;    }

    void display()
    {   inorder(root);
        cout << endl;
    }

    int update_evensum(long long x, node*t) // update evensum, t is current node, x is the value we want insert
    { int newevensum = 0;
      if (abs(x) % 2 == 0){ newevensum = 0; }
      else { newevensum = 1; }
      if (newevensum != t->evensum){t->evensum = 1;}
      else {t->evensum = 0;}
    }
};

vector<long long int> filetoarray(char argv[]){
  //read data to a vector
  ifstream infile;
  infile.open(argv);
  string strLine;
  vector<long long int> data;
  while(getline(infile,strLine))
  { if(strLine.empty())
      continue;
    stringstream ss(strLine);
    cout<<strLine<<endl;

  }
  infile.close();
  return data;
}

int main(int argc, char *argv [])
{
vector<long long int> data,range;
data = filetoarray(argv[1]);


return 0;
}
