#pragma once
#include <iostream>
#include <string>

const std::string jupiter = "/data/sandeep/";

const std::string unity = "/work/spolisetty_umass_edu/data/";

const std::string ornl = "/mnt/bigdata/sandeep/";

const std::string aws  = "/home/ubuntu/data/";

const std::string ERROR = "dir_error";
inline 
std::string get_dataset_dir(){
	char * u = getenv("USER");
	std::string user = std::string(u);
  std::cout << "Found user" << user <<"\n";
  if (user == "spolisetty_umass_edu") return unity;
	if (user == "spolisetty") return jupiter;
  if (user == "q91") return ornl;
  if (user =="root") return aws;
	std::cout << "Datadir not found !!\n";
  if (user == "ubuntu") return aws;
	return ERROR;
}
