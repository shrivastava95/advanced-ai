#include "hmm.cpp"
#include <string>
#include <iostream>
#include <fstream>
using namespace std;
typedef bool (*MyFuncObj)();

/// @brief A struct to run a unit test
class UnitTestRunner
{
public:
    /**
     * @brief Brief description of the tests running
     * in this object
     *
     */
    string TestDescription;
    /**
     * @brief The function to run
     *
     */
    MyFuncObj func;
    /**
     * @brief The unit test to run
     *
     * @return true The unit test has passed
     * @return false The unit test has failed
     */
    bool RunTest()
    {
        if (func != nullptr)
            return func();
        else
            return false;
    }
    UnitTestRunner(const char *desc, MyFuncObj f)
    {
        TestDescription = desc;
        func = f;
    }
};

/// @brief Runs all unit tests and returns the number of unit tests failed
/// @param tests The array of tests to run
/// @param length The number of unit tests
/// @return The number of tests failed
int RunAllTests(UnitTestRunner **tests, int length)
{
    int failed = 0;
    for (int i = 0; i < length; i++)
    {
        cout << "\n\nStarting test #" << (i + 1) << ", '" << tests[i]->TestDescription << "'...\n";
        if (tests[i]->RunTest() == false)
        {
            failed++;
            cout << "Test failed!\n";
        }
        else
        {
            cout << "Test passed!\n";
        }
    }
    return failed;
}
const float epsilon = 0.01;

/// A global instance of the model.
/// All unit tests would be using this model
HMM model;

UnitTestRunner *unit_tests[]{
    // Tests for Question 1
    new UnitTestRunner(
        "Q1: A's Probablity valid test",
        []()
        {
            for (int i = 1; i <= 5; i++)
            {
                float x = 0;
                for (int j = 1; j <= 5; j++)
                {
                    float xi = model.A((SuspectState)i, (SuspectState)j);
                    if (xi < 0 || xi > 1)
                        return false;
                    x += xi;
                }
                x -= 1;
                x *= x;
                if (x > epsilon)
                    return false; // Sum of probablities is not equal to 1
            }
            return true;
        }),
    new UnitTestRunner(
        "Q1: B's Probablity valid test",
        []()
        {
            for (int i = 1; i <= 5; i++)
            {
                float x = 0;
                for (int j = 6; j <= 8; j++)
                {
                    for (int k = 9; k <= 12; k++)
                    {
                        float xi = model.B((SuspectState)i, Observation((Time)j, (Action)k));
                        if (xi < 0 || xi > 1)
                            return false;
                        x += xi;
                    }
                }
                x -= 1;
                x *= x;
                if (x > epsilon)
                    return false; // Sum of probablities is not equal to 1
            }
            return true;
        }),
    new UnitTestRunner(
        "Q1: pi's Probablity valid test",
        []()
        {
            float x = 0;
            for (int j = 1; j <= 5; j++)
            {
                float xi = model.Pi((SuspectState)j);
                if (xi < 0 || xi > 1)
                    return false;
                x += xi;
            }
            x -= 1;
            x *= x;
            if (x > epsilon)
                return false; // Sum of probablities is not equal to 1
            else
                return true;
        }),
    new UnitTestRunner(
        "Q1: Basic HMM assumptions test",
        []()
        {
            double p, q;
            p = model.A(State_Scouting, State_Burglary);
            q = model.A(State_Burglary, State_Scouting);
            return p > q;
        }),
    new UnitTestRunner(
        "Q1: Basic HMM assumptions test",
        []()
        {
            double p, q;
            p = model.A(State_Burglary, State_Migrating);
            q = model.A(State_Migrating, State_Burglary);
            return p > q;
        }),
    new UnitTestRunner(
        "Q1: Basic HMM assumptions test",
        []()
        {
            double p, q, r;
            p = model.B(State_Scouting, Observation(Time_Day, Action_Roaming));
            q = model.B(State_Scouting, Observation(Time_Evening, Action_Roaming));
            r = model.B(State_Scouting, Observation(Time_Night, Action_Roaming));
            return r > q && r > p;
        }),
    new UnitTestRunner(
        "Q1: Basic HMM assumptions test",
        []()
        {
            double p;
            p = model.B(State_Misc, Observation(Time_Evening, Action_Eating));
            return p > 0;
        }),
    new UnitTestRunner(
        "Q1: Basic HMM assumptions test",
        []()
        {
            double p, q;
            p = model.Pi(State_Planning) + model.Pi(State_Misc);
            q = model.Pi(State_Burglary) + model.Pi(State_Scouting) + model.Pi(State_Migrating);
            return p > q;
        }),

    // Tests for question 2

    new UnitTestRunner(
        "Q2: Basic Liklihood Test",
        []()
        {
            Observation o1[6] =
                {
                    Observation(Time_Day, Action_Home),
                    Observation(Time_Evening, Action_Eating),
                    Observation(Time_Night, Action_Home),
                    Observation(Time_Day, Action_Home),
                    Observation(Time_Evening, Action_Eating),
                    Observation(Time_Night, Action_Roaming),
                };
            double p = Liklihood(model, o1, 6);
            return p >= 0 && p <= 1;
        }),
    new UnitTestRunner(
        "Q2: Basic Liklihood Test",
        []()
        {
            Observation o2[6] =
                {
                    Observation(Time_Day, Action_Roaming),
                    Observation(Time_Evening, Action_Roaming),
                    Observation(Time_Night, Action_Untracked),
                    Observation(Time_Day, Action_Untracked),
                    Observation(Time_Evening, Action_Home),
                    Observation(Time_Night, Action_Home),
                };
            double p = Liklihood(model, o2, 6);
            return p >= 0 && p <= 1;
        }),
    new UnitTestRunner(
        "Q2: Simple Comparision Test",
        []()
        {
            Observation o1[6] =
                {
                    Observation(Time_Day, Action_Home),
                    Observation(Time_Evening, Action_Eating),
                    Observation(Time_Night, Action_Home),
                    Observation(Time_Day, Action_Home),
                    Observation(Time_Evening, Action_Eating),
                    Observation(Time_Night, Action_Roaming),
                };
            Observation o2[6] =
                {
                    Observation(Time_Day, Action_Roaming),
                    Observation(Time_Evening, Action_Roaming),
                    Observation(Time_Night, Action_Untracked),
                    Observation(Time_Day, Action_Untracked),
                    Observation(Time_Evening, Action_Home),
                    Observation(Time_Night, Action_Home),
                };
            double p, q;
            p = Liklihood(model, o1, 6);
            q = Liklihood(model, o2, 6);
            return p > q;
        }),

    new UnitTestRunner(
        "Q3: Basic MLE Test",
        []()
        {
            Observation o[] = 
            {
                Observation(Time_Day, Action_Home),
                Observation(Time_Evening, Action_Eating),
                Observation(Time_Night, Action_Home),
                Observation(Time_Day, Action_Home),
                Observation(Time_Evening, Action_Eating),
                Observation(Time_Night, Action_Roaming),
            };
            SuspectState *state = GetHiddenStates(model, o, sizeof(o) / sizeof(o[0]));
            return state != nullptr;
        }),
};

int main()
{
    Observation** database = ReadDataset();
    LearnModel(database, model);
    int result = RunAllTests(unit_tests,
                             sizeof(unit_tests) / sizeof(unit_tests[0]));

    if (result == 0)
        cout << "\n\nAll tests have passed successfully!\n";
    else
        cout << "\n\nSome tests have failed!\n";

    // Memory clean-up
    for(int i=0;i<1000;i++) // Size of dataset is known already
    {
        delete[] database[i];
    }
    delete[] database;

    return result;
}