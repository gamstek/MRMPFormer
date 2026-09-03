#include "json_protocol.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <string_view>

namespace {

void assert_rejected(std::string_view input, std::string_view error_fragment) {
    const ParseResult parsed = parse_input_json(input);
    assert(!parsed.ok);
    assert(parsed.error.find(error_fragment) != std::string::npos);
}

void test_valid_item_is_preserved() {
    const ParseResult parsed = parse_input_json(
        R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"mzq3":99.0,"x":[1.0,1.1],"y":[2.0,3.0]}]})");

    assert(parsed.ok);
    assert(parsed.items.size() == 1);
    assert(parsed.items[0].uid == "c1");
    assert(parsed.items[0].channel == "qual");
    assert(parsed.items[0].mzq1 == 142.0);
    assert(parsed.items[0].mzq3 == 99.0);
    assert(parsed.items[0].x.size() == 2);
    assert(parsed.items[0].y[1] == 3.0);
}

void test_missing_required_fields_are_rejected() {
    assert_rejected(R"({"items":[{"channel":"qual","mzq1":142.0,"mzq3":99.0,"x":[1,2],"y":[2,3]}]})", "items[0].uid");
    assert_rejected(R"({"items":[{"uid":"c1","mzq1":142.0,"mzq3":99.0,"x":[1,2],"y":[2,3]}]})", "items[0].channel");
    assert_rejected(R"({"items":[{"uid":"c1","channel":"qual","mzq3":99.0,"x":[1,2],"y":[2,3]}]})", "items[0].mzq1");
    assert_rejected(R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"x":[1,2],"y":[2,3]}]})", "items[0].mzq3");
    assert_rejected(R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"mzq3":99.0,"y":[2,3]}]})", "items[0].x");
    assert_rejected(R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"mzq3":99.0,"x":[1,2]}]})", "items[0].y");
}

void test_invalid_signal_shapes_are_rejected() {
    assert_rejected(R"([])", "items is required and must be an array");
    assert_rejected(R"({"items":[{"uid":"c1","channel":"qual","mz":142.0,"x":[1,2],"y":[2,3]}]})", "items[0].mz");
    assert_rejected(R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"mzq3":99.0,"x":[1,2,3],"y":[2,3]}]})", "items[0].y");
    assert_rejected(R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"mzq3":99.0,"x":[1],"y":[2]}]})", "items[0].x");
    assert_rejected(R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"mzq3":99.0,"x":[1,2],"y":[2,"NaN"]}]})", "items[0].y[1]");
    assert_rejected(R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"mzq3":99.0,"x":[1,1e999],"y":[2,3]}]})", "input is not valid JSON");
    assert_rejected(R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"mzq3":99.0,"x":[1,1],"y":[2,3]}]})", "items[0].x[1]");
}

void test_nonfinite_optional_smoothing_is_rejected() {
    assert_rejected(
        R"({"items":[{"uid":"c1","channel":"qual","mzq1":142.0,"mzq3":99.0,"smooth_sigma":1e100,"x":[1,2],"y":[2,3]}]})",
        "items[0].smooth_sigma must be finite");
}

void test_output_serialization_has_all_status_shapes() {
    const TaskResult result = {
        {
            {"ok", "ok", {{1.0, 1.2, 0.95}}, {}},
            {"review", "review", {{2.0, 2.2, 0.60}}, {{"review", "SIGNAL_FALLBACK", {}}}},
            {"bad", "alert", {}, {{"fail", "CHANNEL_LOW_INTENSITY", {}}}},
        },
    };

    const std::string output = generate_output_json(result);
    const std::string expected = R"({
  "items": [
    {
      "uid": "ok",
      "status": "ok",
      "peaks": [
        {
          "a": 1.0,
          "b": 1.2,
          "c": 0.95
        }
      ],
      "alerts": []
    },
    {
      "uid": "review",
      "status": "review",
      "peaks": [
        {
          "a": 2.0,
          "b": 2.2,
          "c": 0.6
        }
      ],
      "alerts": [
        {
          "level": "review",
          "code": "SIGNAL_FALLBACK"
        }
      ]
    },
    {
      "uid": "bad",
      "status": "alert",
      "peaks": [],
      "alerts": [
        {
          "level": "fail",
          "code": "CHANNEL_LOW_INTENSITY"
        }
      ]
    }
  ]
})";

    assert(output == expected);
    assert(output.find("\"detail\"") == std::string::npos);
}

}  // namespace

int main() {
    test_valid_item_is_preserved();
    test_missing_required_fields_are_rejected();
    test_invalid_signal_shapes_are_rejected();
    test_nonfinite_optional_smoothing_is_rejected();
    test_output_serialization_has_all_status_shapes();
    return 0;
}
