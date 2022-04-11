#pragma once
#include<string>

struct Pet {
    Pet(const std::string &name);
    void setName(const std::string &name_);
    const std::string &getName() const;

    std::string name;
};
