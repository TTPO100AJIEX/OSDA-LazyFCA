#include <functional>
#include <ranges>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <span>
#include <set>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <utility>

std::vector<std::string> ReadFeatureNames(std::ifstream& datafile)
{
    std::string line; std::getline(datafile, line); std::istringstream splitter(line);
    std::string feature; std::vector<std::string> features;
    while (std::getline(splitter, feature, ',')) features.emplace_back(std::move(feature));
    return features;
}

std::vector<bool> ReadObject(std::string line)
{
    std::string value; std::istringstream splitter(line);
    std::vector<bool> result;
    while (std::getline(splitter, value, ','))
    {
        if (value == "1") result.push_back(true);
        else if (value == "0") result.push_back(false);
        else throw std::runtime_error("Bad input");
    }
    return result;
}

class FCA
{
public:
    FCA(std::vector<std::string> features, std::vector<std::vector<bool>> data)
        : features_(std::move(features)), data_(std::move(data)) {}

    using Objects = std::vector<std::size_t>;
    using Features = std::vector<std::size_t>;

    Features ObjectsPrime(const Objects& objects)
    {
        auto features = (
            std::views::iota(std::size_t(0), features_.size()) |
            std::views::filter([&](const std::size_t feature) {
                return std::ranges::all_of(objects, [&](const std::size_t object) { return data_[object][feature]; });
            })
        );
        Features result;
        for (const std::size_t feature : features) result.push_back(feature);
        return result;
    }
    
    Objects FeaturesPrime(const Features& features)
    {
        auto objects = (
            std::views::iota(std::size_t(0), data_.size()) |
            std::views::filter([&](const std::size_t object) {
                return std::ranges::all_of(features, [&](const std::size_t feature) { return data_[object][feature]; });
            })
        );
        Features result;
        for (const std::size_t object : objects) result.push_back(object);
        return result;
    }

    const std::string& GetFeatureName(const std::size_t& feature) { return features_[feature]; }
    Objects ObjectsClosure(const Objects& objects) { return FeaturesPrime(ObjectsPrime(objects)); }
    Features FeaturesClosure(const Features& features) { return ObjectsPrime(FeaturesPrime(features)); }
    
    using CboCallback = std::function<bool(const Objects&, const Features&)>;
    void Cbo(const CboCallback& callback, const Features& features = {}, const std::size_t add_from = 0)
    {
        const std::set<std::size_t> before_set(features.begin(), features.end());
        const Features closure = FeaturesClosure(features);
        for (const auto& feature : closure)
        {
            if (before_set.contains(feature)) continue;
            if (features.size() != 0 && feature < features[features.size() - 1]) return; // non canonical
        }

        if (callback(FeaturesPrime(closure), closure)) return; // Stop if callback tells so

        const std::set<std::size_t> closure_set(closure.begin(), closure.end());
        for (std::size_t i = add_from; i < features_.size(); ++i)
        {
            if (closure_set.contains(i)) continue;
            Features closure_cpy = closure; closure_cpy.push_back(i);
            Cbo(callback, closure_cpy, i + 1);
        }
    }

private:
    std::vector<std::string> features_;
    std::vector<std::vector<bool>> data_;
};

bool DoesMatch(const std::vector<bool>& object, const FCA::Features& features)
{
    return std::ranges::all_of(features, [&](const std::size_t feature) { return object[feature]; });
}

std::size_t GetNumMatches(
    const std::vector<std::vector<bool>>& objects,
    const FCA::Features& features
) {
    return std::ranges::count_if(objects, [&](const std::vector<bool>& object) { return DoesMatch(object, features); });
}

using IsClassifierChecker = std::function<bool(const std::size_t&, const std::size_t&)>;
void FindHypotheses(
    const std::vector<std::string>& features,
    const std::vector<std::vector<bool>>& supporters,
    const std::vector<std::vector<bool>>& opposers,
    const std::string& to_file,
    const IsClassifierChecker& is_classifier_checker,
    const std::size_t objects_threshold
) {
    FCA fca(features, supporters);

    std::ofstream output(to_file);
    auto callback = [&](const FCA::Objects& objects, const FCA::Features& features) -> bool {
        if (objects.size() < objects_threshold) return true;

        const std::size_t num_positive = GetNumMatches(supporters, features);
        const std::size_t num_negative = GetNumMatches(opposers, features);
        const bool is_classifier = is_classifier_checker(num_positive, num_negative);

        if (is_classifier)
        {
            output << "({";
            for (std::size_t i = 0; i < features.size(); i++)
            {
                output << fca.GetFeatureName(features[i]);
                if (i != features.size() - 1) output << ',';
            }
            output << "}, {";
            for (std::size_t i = 0; i < objects.size(); i++)
            {
                output << objects[i] + 1;
                if (i != objects.size() - 1) output << ',';
            }
            output << "})\n";
        }
        return is_classifier;
    };

    fca.Cbo(callback);
}


int main()
{
    // std::ios::sync_with_stdio(false);
    std::string line;

    std::ifstream pos_file("../positive.csv");
    const std::vector<std::string> features = ReadFeatureNames(pos_file);
    std::vector<std::vector<bool>> pos_data;
    while (std::getline(pos_file, line)) pos_data.push_back(ReadObject(line));

    std::ifstream neg_file("../negative.csv");
    ReadFeatureNames(neg_file);
    std::vector<std::vector<bool>> neg_data;
    while (std::getline(neg_file, line)) neg_data.push_back(ReadObject(line));

    const auto is_positive_classifier = [](const std::size_t num_positive, const std::size_t num_negative) {
        return 2.75 * num_positive > num_negative && num_positive >= 5;
    };
    const auto is_negative_classifier = [](const std::size_t num_negative, const std::size_t num_positive) {
        return 0.25 * num_negative > num_positive && num_negative >= 10;
    };

    FindHypotheses(features, pos_data, neg_data, "../positive_hypotheses.txt", is_positive_classifier, 400);
    std::cout << "Positive complete" << std::endl;
    FindHypotheses(features, neg_data, pos_data, "../negative_hypotheses.txt", is_negative_classifier, 1150);
}
