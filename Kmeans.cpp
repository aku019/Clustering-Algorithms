#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <fstream>

using namespace std;

class Point
{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_values;
	string name;

public:
	
	Point(int id_point, vector<double>& values, string name = "")
	{
		this->id_point = id_point;
		total_values = values.size();

		for(int i = 0; i < total_values; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

	int getID()
	{
		return id_point;
	}

	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

	int getCluster()
	{
		return id_cluster;
	}

	double getValue(int index)
	{
		return values[index];
	}

	int getTotalValues()
	{
		return total_values;
	}

	void addValue(double value)
	{
		values.push_back(value);
	}

	string getName()
	{
		return name;
	}
};

class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;
	vector<Point> points;

public:
	Cluster(int id_cluster, Point point)
	{
		this->id_cluster = id_cluster;

		int total_values = point.getTotalValues();

		for(int i = 0; i < total_values; i++)
			central_values.push_back(point.getValue(i));

		points.push_back(point);
	}

	void addPoint(Point point)
	{
		points.push_back(point);
	}

	bool removePoint(int id_point)
	{
		int total_points = points.size();

		for(int i = 0; i < total_points; i++)
		{
			if(points[i].getID() == id_point)
			{
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	Point getPoint(int index)
	{
		return points[index];
	}

	int getTotalPoints()
	{
		return points.size();
	}

	int getID()
	{
		return id_cluster;
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

	// return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		for(int i = 0; i < total_values; i++)
		{
			sum += pow(clusters[0].getCentralValue(i) -
					   point.getValue(i), 2.0);
		}

		min_dist = sqrt(sum);

		for(int i = 1; i < K; i++)
		{
			double dist;
			sum = 0.0;

			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[i].getCentralValue(j) -
						   point.getValue(j), 2.0);
			}

			dist = sqrt(sum);

			if(dist < min_dist)
			{
				min_dist = dist;
				id_cluster_center = i;
			}
		}

		return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	void run(vector<Point> & points)
	{
		if(K > total_points)
			return;

		vector<int> prohibited_indexes;

		// choose K distinct values for the centers of the clusters
		for(int i = 0; i < K; i++)
		{
			while(true)
			{
				int index_point = rand() % total_points;

				if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
						index_point) == prohibited_indexes.end())
				{
					prohibited_indexes.push_back(index_point);
					points[index_point].setCluster(i);
					Cluster cluster(i, points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
		}

		int iter = 1;

		while(true)
		{
			bool done = true;

			// associates each point to the nearest center
			for(int i = 0; i < total_points; i++)
			{
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);

				if(id_old_cluster != id_nearest_center)
				{
					if(id_old_cluster != -1)
						clusters[id_old_cluster].removePoint(points[i].getID());

					points[i].setCluster(id_nearest_center);
					clusters[id_nearest_center].addPoint(points[i]);
					done = false;
				}
			}

			// recalculating the center of each cluster
			for(int i = 0; i < K; i++)
			{
				for(int j = 0; j < total_values; j++)
				{
					int total_points_cluster = clusters[i].getTotalPoints();
					double sum = 0.0;

					if(total_points_cluster > 0)
					{
						for(int p = 0; p < total_points_cluster; p++)
							sum += clusters[i].getPoint(p).getValue(j);
						clusters[i].setCentralValue(j, sum / total_points_cluster);
					}
				}
			}

			if(iter >= max_iterations)
			{
				cout << "Break in iteration " << iter << "\n\n";
				break;
			}

			iter++;
		}

		

		// shows elements of clusters
		for(int i = 0; i < K; i++)
		{
			int total_points_cluster =  clusters[i].getTotalPoints();
			cout<<total_points_cluster<<"\n";

			int c=0;

			cout << "Cluster " << clusters[i].getID() + 1 << endl;
			
			for(int j = 0; j < total_points_cluster; j++)
			{
				cout<<"[";
				//\cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
				for(int p = 0; p < total_values; p++)
				{
					cout << clusters[i].getPoint(j).getValue(p) ;
					if(p==0)
					{
						cout<<",";
						c++;
					}


					
				}
				cout<<"],";

				string point_name = clusters[i].getPoint(j).getName();

				if(point_name != "")
					cout << "- " << point_name;

				cout << endl;
			}

			cout << "Cluster values: ";
			cout<<"[";

			for(int j = 0; j < total_values; j++)
			{
				
				cout << clusters[i].getCentralValue(j) << ",";
			}

			cout<<"]";

			cout << "\n\n";
		}
	}

	 
};

int main(int argc, char *argv[])
{
	srand (time(NULL));
	int c=0;

	int total_points, total_values, K, max_iterations, has_name;
     //total_points=7;
     //K=2;
     //max_iterations=10;
     //total_values=2;
	
	    //vector<vector<double>> d= {{1.0,1.0},{1.5,2.0},{3.0,4.0},{5.0,7.0},{3.5,5.0},{4.5,5.0},{3.5,4.5}};
	vector<Point> points;
	string point_name;
	
    //for(int i = 0; i < total_points; i++)
	//{
	//	vector<double> values;

	//	for(int j = 0; j < total_values; j++)
	//	{
	//		double value;
	//		
	//		value=d[i][j];
	//		values.push_back(value);
	//	}



		
			//Point p(i, values);
			//points.push_back(p);
		
	//}
	//char ch;


	
	//{
	//	ip.get(ch);
		
	//	if(ch=='1'||ch=='2'||ch==',')
	//	{
	//		c++;
	//		continue;
	//	}
	//	else
	//		values.push_back(ch);
//
//	}
	
	//Point p(i, values);
	//points.push_back(p);
	
	ifstream ip("Image11.txt");
	
	
	string id1;
	string id2;
	string id3;
	double i1,i2;
	if(ip.is_open())
	{
		
		
		while(ip)
		{
			vector<double> values;
			getline(ip,id1,',');
			getline(ip,id2,',');
			getline(ip,id3,'\n');

			 //cout<<id1<<"  "<<id2<<"\n";

			
		  
		    c++;
		    

		   i1 = atof(id1.c_str());
		   i2 = atof(id2.c_str());
		   //cout<<i1<<"  "<<i2<<"\n";



		    values.push_back(i1);
		    values.push_back(i2);
		    


		 
		    Point p(c, values);
			points.push_back(p);
			//if(c==7)
			//{
				//cout<<"First one\n";
				//for(int j=0;j<2*c;j++)
					//{
					//	cout<<p.values[j];
					//	cout<<"\n";
					//}
			//	break;
			
		
			//}
		

		}
		//cout<<"it begins\n";

		//for(int i=0;i<2*c;i++)
		//{
			
		//	cout<<values[i]<<"\n";
		//}
		
		//cout<<"\n"<<"again\n";





		//while(ip>> id1>> id2 >> id3)
		//{
		//	c++;
		//	cout<< id1<<","<<id2<<","<<id3<<"\n";
		//	cout<<c<<"\n";
	

	}

	else
		cout<<" File not there\n";
	
	ip.close();


      total_points = c;
	K=10;
	max_iterations=2000;
	total_values=2;



	KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run(points);
	cout<<total_points;

	return 0;
}