/// @brief All possible states of the suspect
enum SuspectState
{
    State_Planning = 1,
    State_Scouting = 2,
    State_Burglary = 3,
    State_Migrating = 4,
    State_Misc = 5,
};

/// @brief All possible time slots
enum Time
{
    Time_Day = 6,
    Time_Evening = 7,
    Time_Night = 8,
};

/// @brief All possible observations done for suspect
enum Action
{
    Action_Roaming = 9,
    Action_Eating = 10,
    Action_Home = 11,
    Action_Untracked = 12,
};

/// @brief Time and Action is wrapped in a struct
struct Observation
{
    Time time;
    Action action;
    Observation() {}
    Observation(Time t, Action a)
    {
        time = t;
        action = a;
    }
};

#include<fstream>
/// @brief Reads the dataset of array of sequence of observation
/// @return array of sequence of observation
Observation **ReadDataset()
{
    std::ifstream file("database.txt", std::ios_base::in); // filename of the dataset is hardcoded here
    int p;
    file >> p; // The size of outer array (The number of sequences of sequences)
    Observation **Data = new Observation *[p];
    for (int i = 0; i < p; i++)
    {
        int q;
        file >> q;
        Data[i] = new Observation[q];
        int d, a;
        for (int j = 0; j < q; j++)
        {
            Time dt;
            Action at;
            file >> d >> a;
            dt = (Time)d;
            at = (Action)a;
            Data[i][j] = Observation(dt, at);
        }
    }
    file.close();
    return Data;
}

//--------------Do not change anything above this line---------------

class HMM
{
    // Add more code here if needed
public:
    double A(SuspectState a, SuspectState b)
    {
        // Complete the code to return the output
        // of transition probablity to going
        // from state 'a' to state 'b'
        // Hint: The neccessary code does not need
        // to be only within this function
    }
    double B(SuspectState a, Observation b)
    {
        // Complete the code to return the output
        // of probablity of getting observation
        // from state 'b' at state 'a'
        // Hint: The neccessary code does not need
        // to be only within this function
    }
    double Pi(SuspectState a)
    {
        // Complete the code to return the
        // probablity of starting from this
        // state 'a'
        // Hint: The neccessary code does not need
        // to be only within this function
    }
};

// Part I
//---------

/// @brief Reads the dataset of array of sequence of observation
/// and initializes a HMM model from it
/// @param dataset The file to read the observation sequence from
/// @param model The model to learn. Note that it is passed as reference
void LearnModel(Observation **dataset, HMM &model)
{
    // Complete this function
}

// Part II
//---------

/// @brief Given an initialized HMM model,
/// and some set of observations, this function evaluates
/// the liklihood that this set of observation was indeed
/// generated from the given model
/// @param hmm_model The given HMM model
/// @param o The given set of observations
/// @param count The count of the observations
/// @return The probablity/liklihood of this observation
double Liklihood(const HMM &hmm_model, const Observation *o, const int count)
{
    // Complete the function
}

// Part III
//---------

/// @brief Given an initialized model, and a sequence of observation, returns
/// a newly allocated array of the same size, which contains the most likely
/// states the model was in to produce the given observations
/// @param model The initialized model
/// @param o The array/sequence of observations
/// @param size The size of the array of observation
/// @return An array/sequence of states that the model was in to
/// produce the corresponding sequence of observations
SuspectState *GetHiddenStates(const HMM &model, const Observation *o, const int size)
{
    // Complete the function
}