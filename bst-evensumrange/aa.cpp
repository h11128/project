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
    {
        long long data;
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
            t->evensum = 0;
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

    int isempty()
    { if (root == NULL){return 1;}  }

    void insert(long long x)
    {        root = insert(x, root);    }

    int evenSumRange()
    {        return root->evensum;    }

    void display()
    {   inorder(root);
        cout << endl;
    }

    int update_evensum(long long x, node*t) // update evensum, t is current node, x is the value we want insert
    { //cout<<t->data<<" :evensum from "<<t->evensum<<" update to ";
      int newevensum = 0;
      if (abs(x) % 2 == 0){ newevensum = 0; }
      else { newevensum = 1; }
      if (newevensum != t->evensum){t->evensum = 1;}
      else {t->evensum = 0;}
      //cout<<t->evensum<<endl;
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
    int temp;
    while(ss >> temp){data.push_back(temp);//cout<<temp<<" ";
  }
  }
  //cout<<endl;
  infile.close();
  return data;
}

int main(int argc, char *argv [])
{
vector<long long int> data,range;
data = filetoarray(argv[1]);
range = filetoarray(argv[2]);

for( int j = 0; j<range.size(); j = j+2)
{ BST t; // construct a new tree everytime
  //cout<<"Range ["<<range[j]<<","<<range[j+1]<<"]: insert ";
  for( int k = 0; k<data.size(); k = k+1)
  {
    if (data[k] >= range[j] and data[k] <= range[j+1]) // insert data to tree t in this range
    {    //cout<<data[k]<<" ";
        t.insert(data[k]);
      }
  }
  //cout<<endl;
  if ( t.isempty() == 1){ cout<<"Range ["<<range[j]<<","<<range[j+1]<<"]: not number in the range, even sum is NULL!"<<endl;  }
  else if ( t.evenSumRange() == 0)
  {  cout<<"Range ["<<range[j]<<","<<range[j+1]<<"]: even sum"<<endl;  }
  else {cout<<"Range ["<<range[j]<<","<<range[j+1]<<"]: odd sum"<<endl;}
  //t.display();
}

return 0;
}
