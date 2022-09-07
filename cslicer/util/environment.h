#pragma once
#include <iostream>
#include <string>


const std::string jupiter = "/data/sandeep/";

const std::string unity = "/work/spolisetty_umass_edu/data/";

const std::string ornl = "/tmp/q91";

const std::string ERROR = "dir_error";
std::string get_dataset_dir(){

	char * u = getenv("USER");

	std::string user = std::string(u);
  std::cout << "Found user" << user <<"\n";
  if (user == "spolisetty_umass_edu") return unity;
	if (user == "spolisetty") return jupiter;
  if (user == "q91") return ornl;
	std::cout << "Datadir not found !!\n";
	return ERROR;
}
