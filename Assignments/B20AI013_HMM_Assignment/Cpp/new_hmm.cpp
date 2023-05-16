#include "hmm.cpp" // Observation struct is borrowed here

// Part IV
//---------

/// @brief All possible states of the suspect
enum UpdatedSuspectState
{
    // Add custom states here
};

class Updated_HMM
{
    // Complete this implementation
    // for part IV of the assignment.
};

/// @brief Reads the dataset of array of sequence of observation
/// and initializes a HMM model from it
/// @param dataset The file to read the observation sequence from
/// @param model The model to learn. Note that it is passed as reference
void LearnModel(Observation **dataset, Updated_HMM &model)
{
    // Complete this function
}


/// @brief Given an initialized HMM model,
/// and some set of observations, this function evaluates
/// the liklihood that this set of observation was indeed
/// generated from the given model
/// @param hmm_model The given HMM model
/// @param o The given set of observations
/// @param count The count of the observations
/// @return The probablity/liklihood of this observation
double Liklihood(const Updated_HMM &hmm_model, const Observation *o, const int count)
{
    // Complete the function
}

/// @brief Given an initialized model, and a sequence of observation, returns
/// a newly allocated array of the same size, which contains the most likely
/// states the model was in to produce the given observations
/// @param model The initialized model
/// @param o The array/sequence of observations
/// @param size The size of the array of observation
/// @return An array/sequence of states that the model was in to
/// produce the corresponding sequence of observations
UpdatedSuspectState *GetHiddenStates(const Updated_HMM &model, const Observation *o, const int size)
{
    // Complete the function
}

int main()
{
    Observation** database = ReadDataset(); // Read from "database.txt" file

    HMM old_model;
    Updated_HMM new_model;

    LearnModel(database, old_model);
    LearnModel(database, new_model);

    // Observation list[] = 
    // {
    //     // Add your list of observations
    // };
    // int size = sizeof(list)/sizeof(Observation);
    // double p = Liklihood(old_model, list, size);
    // double q = Liklihood(new_model, list, size);

    // SuspectState* old_states = GetHiddenStates(old_model, list, size);
    // UpdatedSuspectState* new_states = GetHiddenStates(new_model, list, size);
    // .... 
    
    // Add code to showcase and compare the obtained
    // results between the two models


    // Memory clean-up
    for(int i=0;i<1000;i++)
    {
        delete[] database[i];
    }
    delete[] database;
    return 0;
}