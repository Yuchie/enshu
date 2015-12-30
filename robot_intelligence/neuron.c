#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>

/*
  １０ｘ１０画素の濃淡画像を識別（A,B,C,D,E)
  ニューロン数
  　　第一層：100
  　　第二層：150
  　　第三層：5
  結合荷重初期値 平均値0,分散<1が良い
  係数：0.1
  学習則
*/

#define e 0.1 //学習係数
#define layer_1 100 //第一層
#define layer_2 50 //第二層の数
#define layer_3 5 //the third layer
#define noise 0 //ノイズ
#define ndat 1 //訓練数
#define A 0.8 //sigmoidの係数
#define stage 0 /*0:学習データと実験データ（識別データ）が同じ
		  1:学習データと実験データが異なる*/

typedef struct{
  //層
  double l1[layer_1];
  double l2[layer_2];
  double l3[layer_3];
  //結合荷重
  double c12[layer_2][layer_1];
  double c23[layer_3][layer_2];
  //出力
  double l2_e[layer_2];
  double y_e[layer_3];
}_net;

//シグモイド関数
double sigmoid(double u);

//シグモイドの微分
double d_sigmoid(double u);

//出力値の計算
void update_net(double test[10][10], _net *net);

//出力と教師データの誤差
void calc_error(double error[layer_3], int y[layer_3], _net *net);
  
int main(int argc, char **argv)
{
  _net net;
  int i,j,k,l,m;
  double test[ndat][10][10]; //訓練データ
  double test_n[ndat][10][10]; //ノイズ加えた後の訓練データ
  int y[ndat][layer_3]; //出力
  double error[layer_3]; //誤差
  double d_2[layer_2];
  double d_3[layer_3]; //学習時のδ
  double sum=0;
  double cal[layer_3][10][10]; //実験データ
  double cal_n[layer_3][10][10]; //ノイズ加えた後の実験データ
  IplImage *img;
  char data_name[ndat][256]={"A_1.jpg", "A_2.jpg", "A_3.jpg", "A_4.jpg", "A_5.jpg", "A_6.jpg", "A_7.jpg", "A_8.jpg", "A_9.jpg", "A_10.jpg", "B_1.jpg", "B_2.jpg", "B_3.jpg", "B_4.jpg", "B_5.jpg", "B_6.jpg", "B_7.jpg", "B_8.jpg", "B_9.jpg", "B_10.jpg", "C_1.jpg", "C_2.jpg", "C_3.jpg", "C_4.jpg", "C_5.jpg", "C_6.jpg", "C_7.jpg", "C_8.jpg", "C_9.jpg", "C_10.jpg", "D_1.jpg", "D_2.jpg", "D_3.jpg", "D_4.jpg", "D_5.jpg", "D_6.jpg", "D_7.jpg", "D_8.jpg", "D_9.jpg", "D_10.jpg", "E_1.jpg", "E_2.jpg", "E_3.jpg", "E_4.jpg", "E_5.jpg", "E_6.jpg", "E_7.jpg", "E_8.jpg", "E_9.jpg", "E_10.jpg"};
  char calc_name[layer_3][256]={"A_11.jpg", "B_11.jpg", "C_11.jpg", "D_11.jpg", "E_11.jpg"};
  int y_2[layer_3][layer_3];
  
  for(i=0; i<ndat; i++){
    for(j=0; j<layer_3; j++){
      y[i][j]=0;
    }
  }

  for(i=0; i<ndat/5; i++){
    y[i][0]=1;
    y[i+10][1]=1;
    y[i+20][2]=1;
    y[i+30][3]=1;
    y[i+40][4]=1;
  }

  for(i=0; i<layer_3; i++){
    for(j=0; j<layer_3; j++){
      if(i == j){
	y_2[i][j] = 1;
      }else{
	y_2[i][j] = 0;
      }
    }
  }

  //乱数発生のためのシード
  srand((unsigned)time(NULL));
  //結合荷重初期化--ランダムでかつ平均値0,分散<1にしたい
  for(i=0; i<layer_2; i++){
    for(j=0; j<layer_1; j++){
      net.c12[i][j] = (double)rand()/((double)RAND_MAX +1) -0.5;
      while (net.c12[i][j] == 0){
	net.c12[i][j] = (double)rand()/((double)RAND_MAX+1) -0.5;
      }
    }
  }

  for(i=0; i<layer_3; i++){
    for(j=0; j<layer_2; j++){
      net.c23[i][j] = (double)rand()/((double)RAND_MAX +1) -0.5;
      while (net.c23[i][j] == 0){
	net.c23[i][j] = (double)rand()/((double)RAND_MAX+1) -0.5;
      }
    }
  }

  //訓練データを格納
  for (l=0; l<ndat; l++){
    if ((img = cvLoadImage (data_name[l], CV_LOAD_IMAGE_COLOR)) == 0){
      //"img.jpeg"
      printf("%d no file\n", l+1);
      return -1;
    }
    for (i=0; i<img->height; i++){
      for (j=0; j<img->width; j++){
	test[l][i][j]=img->imageData[img->widthStep*i + j*3];
	test_n[l][i][j]=test[l][i][j];
      }
    }
  }

  //実験データを格納
  for (l=0; l<layer_3; l++){
    if ((img = cvLoadImage (calc_name[l], CV_LOAD_IMAGE_COLOR)) == 0){
      printf("%d no file\n", l+ndat+1);
      return -1;
    }
    for(i=0; i<img->height; i++){
      for (j=0;j<img->width; j++){
	cal[l][i][j]=img->imageData[img->widthStep*i +j*3];
	cal_n[l][i][j]=cal[l][i][j];
      }
    }
  }

  for(m=0; m<1000; m++){
  //学習
    if(stage == 0){
      //学習データと実験データが同じ
      //noiseを加える
      for (l=0; l<layer_3; l++){
	for (i=0; i<10; i++){
	  for (j=0; j<10; j++){
	    if(noise!=0){
	      if((rand()%100)<noise){
		cal_n[l][i][j]=rand()%255;
	      }else{
		cal_n[l][i][j]=test[l][i][j];
	      }
	    }
	  }
	}
	update_net(cal_n[l], &net);
	calc_error(error, y_2[l], &net);
	//第二と三層間の重み係数を更新                                        
	for (i=0; i<layer_3; i++){
	  //      printf("error[%d] = %lf \n", i, error[i]);
	  d_3[i] = error[i] * d_sigmoid(net.y_e[i]);
	  for (j=0; j<layer_2; j++){
	    net.c23[i][j] = net.c23[i][j] - e*net.l2_e[j]*d_3[i];
	    //            printf("net.c23[%d][%d] = %lf \n", i, j, net.c23[i][j]);                                                                            
	  }
	}
	//第一と第二層間の重み係数を更新                                
	for (i=0; i<layer_2; i++){
	  sum=0;
	  for (k=0; k<layer_3; k++){
	    sum+=net.c23[k][i]*d_3[k];
	  }
	  d_2[i] = sum*d_sigmoid(net.l2_e[i]);
	  for (j=0; j<layer_1; j++){
	    net.c12[i][j] = net.c12[i][j] - e*net.l1[j]*d_2[i];
	    //            printf("net.c12[%d][%d] = %lf \n", i, j, net.c12[i][j]);                                                                           
	  }
	}	
      }
    }else if(stage == 1){
      //学習データと実験データが異なる
      for (l=0; l<ndat; l++){
	//同様にノイズを加える
	for (i=0; i<10; i++){
          for (j=0; j<10; j++){
            if(noise!=0){
              if((rand()%100)<noise){
                test_n[l][i][j]=rand()%255;
              }else{
		test_n[l][i][j]=test[l][i][j];
	      }
	    }
	  }
	}
	update_net(test_n[l], &net);
	calc_error(error, y[l], &net);
	//第二と三層間の重み係数を更新
	for (i=0; i<layer_3; i++){
	  //      printf("error[%d] = %lf \n", i, error[i]);
	  d_3[i] = error[i] * d_sigmoid(net.y_e[i]);
	  for (j=0; j<layer_2; j++){
	    net.c23[i][j] = net.c23[i][j] - e*net.l2_e[j]*d_3[i];
	    //      	printf("net.c23[%d][%d] = %lf \n", i, j, net.c23[i][j]);
	  }
	}
	//第一と第二層間の重み係数を更新
      for (i=0; i<layer_2; i++){
	sum=0;
	for (k=0; k<layer_3; k++){
	  sum+=net.c23[k][i]*d_3[k];
	}
	d_2[i] = sum*d_sigmoid(net.l2_e[i]);
	for (j=0; j<layer_1; j++){
	  net.c12[i][j] = net.c12[i][j] - e*net.l1[j]*d_2[i];
	  //       	printf("net.c12[%d][%d] = %lf \n", i, j, net.c12[i][j]);
	}
      }
      }
    }
    if (m==100){
      break; //過学習の疑いがあるので100で止めてみる（結局結果は変わらなかった）
    }
  }

  /*  for(i=0; i<layer_2; i++){
    for (j=0; j<layer_1; j++){
      printf("net.c12[%d][%d] = %lf \n", i, j, net.c12[i][j]);
    }
  }
  for(i=0; i<layer_3; i++){
    for(j=0; j<layer_2; j++){
      printf("net.c23[%d][%d] = %lf \n", i, j, net.c23[i][j]);
    }
    }*/

  //学習成果を計算
    for (l=0; l<layer_3; l++){
    update_net(cal[l], &net);
    printf("answer[%d]: %lf %lf %lf %lf %lf\n", l, net.y_e[0], net.y_e[1], net.y_e[2], net.y_e[3], net.y_e[4]);
    }

}


double sigmoid(double u){
  double a;
  a = 1/(1+exp(-A*u));
  return a;
}


double d_sigmoid(double u){
  double a;
  a = A*u*(1-u);
  return a;
}


void update_net(double test[10][10], _net *net){
  int i, j;
  for (i=0; i<10; i++){
    for (j=0; j<10; j++){
      net->l1[i*10+j] = test[i][j];
    }
  }

  for (i=0; i<layer_2; i++){
    net->l2[i] = 0; //初期化
    for(j=0; j<layer_1; j++){
      net->l2[i] += net->c12[i][j] * net->l1[j];
    }
    net->l2_e[i] = sigmoid(net->l2[i]);
  }

  //同様に第三層を計算
  for (i=0; i<layer_3; i++){
    net->l3[i] =0;
    for(j=0; j<layer_2; j++){
      net->l3[i] += net->c23[i][j] * net->l2_e[j];
    }
    net->y_e[i] = sigmoid(net->l3[i]);
  }

}


void calc_error(double error[layer_3], int y[layer_3], _net *net){
  int i;
  for(i=0; i<layer_3; i++){
    error[i] = net->y_e[i]-y[i];
  }

}
