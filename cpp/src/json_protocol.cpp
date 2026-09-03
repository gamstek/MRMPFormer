#include "json_protocol.h"

#include <cmath>
#include <exception>
#include <sstream>

#include <json.hpp>

namespace {

using nlohmann::json;

std::string item_field(std::size_t item_index, std::string_view field) {
    std::ostringstream stream;
    stream << "items[" << item_index << "]." << field;
    return stream.str();
}

ParseResult error_result(std::string error) {
    return {false, {}, std::move(error)};
}

bool read_required_string(const json& item,
                          std::size_t item_index,
                          std::string_view field,
                          std::string* value,
                          std::string* error) {
    const auto found = item.find(field);
    if (found == item.end() || !found->is_string()) {
        *error = item_field(item_index, field) + " is required and must be a string";
        return false;
    }
    *value = found->get<std::string>();
    return true;
}

bool read_required_number(const json& item,
                          std::size_t item_index,
                          std::string_view field,
                          double* value,
                          std::string* error) {
    const auto found = item.find(field);
    if (found == item.end() || !found->is_number()) {
        *error = item_field(item_index, field) + " is required and must be numeric";
        return false;
    }
    *value = found->get<double>();
    if (!std::isfinite(*value)) {
        *error = item_field(item_index, field) + " must be finite";
        return false;
    }
    return true;
}

bool read_signal_array(const json& item,
                       std::size_t item_index,
                       std::string_view field,
                       std::vector<double>* values,
                       std::string* error) {
    const auto found = item.find(field);
    if (found == item.end() || !found->is_array()) {
        *error = item_field(item_index, field) + " is required and must be an array";
        return false;
    }

    values->clear();
    values->reserve(found->size());
    for (std::size_t value_index = 0; value_index < found->size(); ++value_index) {
        const json& value = (*found)[value_index];
        if (!value.is_number()) {
            *error = item_field(item_index, field) + "[" + std::to_string(value_index) +
                "] must be numeric and finite";
            return false;
        }
        const double number = value.get<double>();
        if (!std::isfinite(number)) {
            *error = item_field(item_index, field) + "[" + std::to_string(value_index) +
                "] must be finite";
            return false;
        }
        values->push_back(number);
    }
    return true;
}

}  // namespace

ParseResult parse_input_json(std::string_view input) {
    try {
        const json root = json::parse(input.begin(), input.end());
        const auto items_it = root.find("items");
        if (!root.is_object() || items_it == root.end() || !items_it->is_array()) {
            return error_result("items is required and must be an array");
        }

        ParseResult result;
        result.ok = true;
        result.items.reserve(items_it->size());
        for (std::size_t item_index = 0; item_index < items_it->size(); ++item_index) {
            const json& item = (*items_it)[item_index];
            if (!item.is_object()) {
                return error_result("items[" + std::to_string(item_index) + "] must be an object");
            }
            if (item.contains("mz")) {
                return error_result(item_field(item_index, "mz") + " is not supported; use mzq1 and mzq3");
            }

            CompoundData compound;
            std::string error;
            if (!read_required_string(item, item_index, "uid", &compound.uid, &error) ||
                !read_required_string(item, item_index, "channel", &compound.channel, &error) ||
                !read_required_number(item, item_index, "mzq1", &compound.mzq1, &error) ||
                !read_required_number(item, item_index, "mzq3", &compound.mzq3, &error) ||
                !read_signal_array(item, item_index, "x", &compound.x, &error) ||
                !read_signal_array(item, item_index, "y", &compound.y, &error)) {
                return error_result(std::move(error));
            }
            if (compound.channel.empty()) {
                return error_result(item_field(item_index, "channel") + " must not be empty");
            }
            if (compound.x.size() != compound.y.size()) {
                return error_result(item_field(item_index, "y") + " must have the same length as x");
            }
            if (compound.x.size() < 2) {
                return error_result(item_field(item_index, "x") + " and y must contain at least two points");
            }
            for (std::size_t point_index = 1; point_index < compound.x.size(); ++point_index) {
                if (compound.x[point_index] <= compound.x[point_index - 1]) {
                    return error_result(item_field(item_index, "x") + "[" +
                                        std::to_string(point_index) + "] must be strictly increasing");
                }
            }

            if (const auto name = item.find("name"); name != item.end()) {
                if (!name->is_string()) {
                    return error_result(item_field(item_index, "name") + " must be a string");
                }
                compound.name = name->get<std::string>();
            }
            if (const auto sigma = item.find("smooth_sigma"); sigma != item.end()) {
                if (!sigma->is_number()) {
                    return error_result(item_field(item_index, "smooth_sigma") + " must be numeric");
                }
                const double value = sigma->get<double>();
                if (!std::isfinite(value)) {
                    return error_result(item_field(item_index, "smooth_sigma") + " must be finite");
                }
                const float smooth_sigma = static_cast<float>(value);
                if (!std::isfinite(smooth_sigma)) {
                    return error_result(item_field(item_index, "smooth_sigma") + " must be finite");
                }
                compound.smooth_sigma = smooth_sigma;
            }

            result.items.push_back(std::move(compound));
        }
        return result;
    } catch (const std::exception& exception) {
        return error_result(std::string("input is not valid JSON: ") + exception.what());
    }
}

std::string generate_output_json(const TaskResult& result) {
    nlohmann::ordered_json output;
    output["items"] = nlohmann::ordered_json::array();
    for (const CompoundResult& item : result.items) {
        nlohmann::ordered_json serialized_item;
        serialized_item["uid"] = item.uid;
        serialized_item["status"] = item.status;
        serialized_item["peaks"] = nlohmann::ordered_json::array();
        for (const PeakResult& peak : item.peaks) {
            nlohmann::ordered_json serialized_peak;
            serialized_peak["a"] = peak.a;
            serialized_peak["b"] = peak.b;
            serialized_peak["c"] = peak.c;
            serialized_item["peaks"].push_back(std::move(serialized_peak));
        }

        serialized_item["alerts"] = nlohmann::ordered_json::array();
        for (const Alert& alert : item.alerts) {
            nlohmann::ordered_json serialized_alert;
            serialized_alert["level"] = alert.level;
            serialized_alert["code"] = alert.code;
            if (!alert.detail.empty()) {
                nlohmann::ordered_json serialized_detail;
                for (const auto& [key, value] : alert.detail) {
                    serialized_detail[key] = value;
                }
                serialized_alert["detail"] = std::move(serialized_detail);
            }
            serialized_item["alerts"].push_back(std::move(serialized_alert));
        }
        output["items"].push_back(std::move(serialized_item));
    }
    return output.dump(2);
}
