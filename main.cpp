#include <iostream>
#include <string>
#include <cctype>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <regex>
#include <fstream>
#include <memory>
#include <chrono>
#include <thread>
#include <future>
#include <filesystem>
#include <variant>
#include <optional>
#include <random>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <functional>

namespace fs = std::filesystem;

class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(const std::string& msg) = 0;
};

class ConsoleLogger : public ILogger {
    std::mutex mtx;
public:
    void log(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[LOG] " << msg << "\n";
    }
};

class FileLogger : public ILogger {
    std::mutex mtx;
    std::ofstream file;
public:
    explicit FileLogger(const std::string& filename) : file(filename, std::ios::app) {
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open log file: " + filename + " (check permissions or path)");
        }
    }
    ~FileLogger() override { if (file.is_open()) file.close(); }
    void log(const std::string& msg) override {
        std::lock_guard<std::mutex> lock(mtx);
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::string timestamp = std::ctime(&time);
        timestamp.pop_back();
        file << "[" << timestamp << "] " << msg << std::endl;
    }
};

class MultiLogger : public ILogger {
    std::vector<std::unique_ptr<ILogger>> loggers;
public:
    void addLogger(std::unique_ptr<ILogger> logger) { loggers.push_back(std::move(logger)); }
    void log(const std::string& msg) override {
        for (auto& logger : loggers) logger->log(msg);
    }
};

struct ExpressionNode {
    enum class Type { Number, Variable, Operator, Function };
    Type type;
    std::variant<double, std::string, char, std::pair<std::string, std::shared_ptr<ExpressionNode>>> value;
    std::vector<std::shared_ptr<ExpressionNode>> children;

    ExpressionNode(Type t, const std::variant<double, std::string, char, std::pair<std::string, std::shared_ptr<ExpressionNode>>>& v)
        : type(t), value(v) {}

    std::string to_string() const {
        switch (type) {
            case Type::Number: return std::to_string(std::get<double>(value));
            case Type::Variable: return std::get<std::string>(value);
            case Type::Operator: {
                auto op = std::get<char>(value);
                if (children.size() != 2) return "INVALID_OP";
                return "(" + children[0]->to_string() + " " + op + " " + children[1]->to_string() + ")";
            }
            case Type::Function: {
                auto [name, arg] = std::get<std::pair<std::string, std::shared_ptr<ExpressionNode>>>(value);
                return name + "(" + arg->to_string() + ")";
            }
        }
        return "INVALID";
    }
};

class MathPlugin {
public:
    virtual ~MathPlugin() = default;
    virtual std::string name() const = 0;
    virtual double apply(double arg) const = 0;
    virtual std::optional<std::string> simplify(const std::shared_ptr<ExpressionNode>& expr) const { return std::nullopt; }
};

class IntegralPlugin : public MathPlugin {
public:
    std::string name() const override { return "integral"; }
    double apply(double arg) const override {
        if (arg < 0) throw std::runtime_error("Integral domain error");
        return arg * arg / 2;
    }
    std::optional<std::string> simplify(const std::shared_ptr<ExpressionNode>& expr) const override {
        if (!expr) return std::nullopt;
        return "∫(" + expr->to_string() + ")dx";
    }
};

class DerivativePlugin : public MathPlugin {
public:
    std::string name() const override { return "derivative"; }
    double apply(double arg) const override { return arg; }
    std::optional<std::string> simplify(const std::shared_ptr<ExpressionNode>& expr) const override {
        if (!expr) return std::nullopt;
        return "d/dx(" + expr->to_string() + ")";
    }
};

class ScriptEngine {
    std::unordered_map<std::string, std::function<double(double)>> scripts;
    mutable std::mutex mtx;
    std::shared_ptr<ILogger> logger;

public:
    explicit ScriptEngine(std::shared_ptr<ILogger> log = nullptr) : logger(log) {}

    void define(const std::string& name, const std::string& code) {
        std::lock_guard<std::mutex> lock(mtx);
        if (code.find("return ") == 0) {
            std::string expr = code.substr(7);
            expr.erase(0, expr.find_first_not_of(" \t"));
            expr.erase(expr.find_last_not_of(" \t") + 1);

            if (expr == "x * 2") scripts[name] = [](double x) { return x * 2; };
            else if (expr == "x + 1") scripts[name] = [](double x) { return x + 1; };
            else if (expr == "x * x" || expr == "x^2") scripts[name] = [](double x) { return x * x; };
            else if (expr == "sin(x)") scripts[name] = [](double x) { return std::sin(x); };
            else if (expr == "cos(x)") scripts[name] = [](double x) { return std::cos(x); };
            else if (expr == "sqrt(x)") scripts[name] = [](double x) {
                if (x < 0) throw std::runtime_error("Square root of negative number");
                return std::sqrt(x);
            };
            else if (expr == "1/x" || expr == "x^(-1)") scripts[name] = [](double x) {
                if (x == 0) throw std::runtime_error("Division by zero");
                return 1.0 / x;
            };
            else throw std::runtime_error("Unsupported script expression: " + expr);

            if (logger) logger->log("Script '" + name + "' defined with expression: " + expr);
        } else {
            throw std::runtime_error("Script must start with 'return '");
        }
    }

    std::optional<double> execute(const std::string& name, double arg) {
        std::lock_guard<std::mutex> lock(mtx);
        if (auto it = scripts.find(name); it != scripts.end()) {
            try {
                double result = it->second(arg);
                if (logger) logger->log("Script '" + name + "' executed with arg " + std::to_string(arg) + " = " + std::to_string(result));
                return result;
            } catch (const std::exception& e) {
                if (logger) logger->log("Script '" + name + "' execution failed: " + e.what());
                throw;
            }
        }
        return std::nullopt;
    }

    bool exists(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mtx);
        return scripts.find(name) != scripts.end();
    }

    std::vector<std::string> listScripts() const {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<std::string> result;
        for (const auto& [name, _] : scripts) result.push_back(name);
        return result;
    }
};

class SmartMathInterpreter {
private:
    std::string input;
    size_t pos = 0;
    char current_char = '\0';
    std::shared_ptr<ILogger> logger;
    std::unordered_map<std::string, double> variables;
    std::vector<std::tuple<std::string, std::variant<double, std::string>, std::string>> history;
    std::unordered_map<std::string, std::unique_ptr<MathPlugin>> plugins;
    ScriptEngine script_engine;
    std::unordered_map<std::string, double> ai_weights;
    std::mutex mtx;
    std::condition_variable cv;
    bool symbolic_mode = false, parallel_mode = false, debug_mode = false;
    double precision = 1e-10;
    std::string version = "5.0.1";

    void initialize() {
        plugins["integral"] = std::make_unique<IntegralPlugin>();
        plugins["derivative"] = std::make_unique<DerivativePlugin>();
        variables["pi"] = M_PI;
        variables["e"] = M_E;
        logger->log("SmartMathInterpreter v" + version + " initialized");
    }

    void advance() { pos++; current_char = (pos < input.length()) ? input[pos] : '\0'; }
    void skip_whitespace() { while (current_char != '\0' && std::isspace(current_char)) advance(); }

    double parse_number() {
        std::string num_str;
        bool has_decimal = false;
        if (current_char == '-' || current_char == '+') { num_str += current_char; advance(); }
        while (current_char != '\0' && std::isdigit(current_char)) { num_str += current_char; advance(); }
        if (current_char == '.') {
            has_decimal = true;
            num_str += current_char;
            advance();
            if (!std::isdigit(current_char)) throw std::runtime_error("Expected digit after decimal point");
            while (current_char != '\0' && std::isdigit(current_char)) { num_str += current_char; advance(); }
        }
        if (current_char == 'e' || current_char == 'E') {
            num_str += current_char;
            advance();
            if (current_char == '+' || current_char == '-') { num_str += current_char; advance(); }
            if (!std::isdigit(current_char)) throw std::runtime_error("Expected digit in exponent");
            while (current_char != '\0' && std::isdigit(current_char)) { num_str += current_char; advance(); }
        }
        if (num_str.empty() || num_str == "-" || num_str == "+" || num_str == ".") throw std::runtime_error("Invalid number: " + num_str);
        try { return std::stod(num_str); } catch (const std::exception& e) { throw std::runtime_error("Invalid number format: " + num_str); }
    }

    std::string parse_identifier() {
        std::string id;
        if (!(std::isalpha(current_char) || current_char == '_')) throw std::runtime_error("Identifier must start with a letter or underscore");
        while (current_char != '\0' && (std::isalnum(current_char) || current_char == '_')) { id += current_char; advance(); }
        if (id.empty()) throw std::runtime_error("Empty identifier");
        std::lock_guard<std::mutex> lock(mtx);
        ai_weights[id] += 1.0;
        return id;
    }

    std::shared_ptr<ExpressionNode> parse_symbolic_factor() {
        skip_whitespace();
        if (current_char == '(') {
            advance();
            auto expr = parse_symbolic_expression();
            if (current_char != ')') throw std::runtime_error("Missing ')' at position " + std::to_string(pos));
            advance();
            return expr;
        }
        if (std::isdigit(current_char) || current_char == '-' || current_char == '+' || current_char == '.') {
            return std::make_shared<ExpressionNode>(ExpressionNode::Type::Number, parse_number());
        }
        if (std::isalpha(current_char) || current_char == '_') {
            std::string id = parse_identifier();
            skip_whitespace();
            if (current_char == '(') {
                advance();
                auto arg = parse_symbolic_expression();
                if (current_char != ')') throw std::runtime_error("Missing ')' in function " + id);
                advance();
                return std::make_shared<ExpressionNode>(ExpressionNode::Type::Function, std::make_pair(id, arg));
            }
            return std::make_shared<ExpressionNode>(ExpressionNode::Type::Variable, id);
        }
        throw std::runtime_error("Invalid symbolic factor at position " + std::to_string(pos));
    }

    std::shared_ptr<ExpressionNode> parse_symbolic_power() {
        auto base = parse_symbolic_factor();
        skip_whitespace();
        while (current_char == '^') {
            advance();
            auto exp = parse_symbolic_factor();
            auto node = std::make_shared<ExpressionNode>(ExpressionNode::Type::Operator, '^');
            node->children = {base, exp};
            base = node;
            skip_whitespace();
        }
        return base;
    }

    std::shared_ptr<ExpressionNode> parse_symbolic_term() {
        auto result = parse_symbolic_power();
        skip_whitespace();
        while (current_char == '*' || current_char == '/' || current_char == '%') {
            char op = current_char;
            advance();
            auto right = parse_symbolic_power();
            auto node = std::make_shared<ExpressionNode>(ExpressionNode::Type::Operator, op);
            node->children = {result, right};
            result = node;
            skip_whitespace();
        }
        return result;
    }

    std::shared_ptr<ExpressionNode> parse_symbolic_expression() {
        auto result = parse_symbolic_term();
        skip_whitespace();
        while (current_char == '+' || current_char == '-') {
            char op = current_char;
            advance();
            auto right = parse_symbolic_term();
            auto node = std::make_shared<ExpressionNode>(ExpressionNode::Type::Operator, op);
            node->children = {result, right};
            result = node;
            skip_whitespace();
        }
        return result;
    }

    std::shared_ptr<ExpressionNode> simplify(std::shared_ptr<ExpressionNode> expr) {
        if (!expr || expr->children.empty()) return expr;
        if (expr->type == ExpressionNode::Type::Operator) {
            expr->children[0] = simplify(expr->children[0]);
            expr->children[1] = simplify(expr->children[1]);
            char op = std::get<char>(expr->value);
            if (op == '+' && expr->children[0]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[0]->value)) < precision) {
                return expr->children[1];
            }
            if (op == '+' && expr->children[1]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[1]->value)) < precision) {
                return expr->children[0];
            }
            if (op == '-' && expr->children[1]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[1]->value)) < precision) {
                return expr->children[0];
            }
            if (op == '*' && ((expr->children[0]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[0]->value)) < precision) ||
                (expr->children[1]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[1]->value)) < precision))) {
                return std::make_shared<ExpressionNode>(ExpressionNode::Type::Number, 0.0);
            }
            if (op == '*' && expr->children[0]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[0]->value) - 1.0) < precision) {
                return expr->children[1];
            }
            if (op == '*' && expr->children[1]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[1]->value) - 1.0) < precision) {
                return expr->children[0];
            }
            if (op == '/' && expr->children[0]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[0]->value)) < precision) {
                return std::make_shared<ExpressionNode>(ExpressionNode::Type::Number, 0.0);
            }
            if (op == '/' && expr->children[1]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[1]->value) - 1.0) < precision) {
                return expr->children[0];
            }
            if (op == '^' && expr->children[1]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[1]->value)) < precision) {
                return std::make_shared<ExpressionNode>(ExpressionNode::Type::Number, 1.0);
            }
            if (op == '^' && expr->children[1]->type == ExpressionNode::Type::Number && 
                std::abs(std::get<double>(expr->children[1]->value) - 1.0) < precision) {
                return expr->children[0];
            }
            if (expr->children[0]->type == ExpressionNode::Type::Number && 
                expr->children[1]->type == ExpressionNode::Type::Number) {
                double left = std::get<double>(expr->children[0]->value);
                double right = std::get<double>(expr->children[1]->value);
                double result;
                switch (op) {
                    case '+': result = left + right; break;
                    case '-': result = left - right; break;
                    case '*': result = left * right; break;
                    case '/': 
                        if (std::abs(right) < precision) throw std::runtime_error("Division by zero during simplification");
                        result = left / right; 
                        break;
                    case '^': result = std::pow(left, right); break;
                    case '%': 
                        if (std::abs(right) < precision) throw std::runtime_error("Modulo by zero during simplification");
                        result = std::fmod(left, right); 
                        break;
                    default: throw std::runtime_error("Unknown operator during simplification");
                }
                return std::make_shared<ExpressionNode>(ExpressionNode::Type::Number, result);
            }
        } else if (expr->type == ExpressionNode::Type::Function) {
            auto pair = std::get<std::pair<std::string, std::shared_ptr<ExpressionNode>>>(expr->value);
            std::string name = pair.first;
            std::shared_ptr<ExpressionNode> arg = pair.second;
            arg = simplify(arg);
            expr->value = std::make_pair(name, arg);
            if (plugins.count(name)) {
                if (auto simplified = plugins[name]->simplify(arg); simplified) {
                    return std::make_shared<ExpressionNode>(ExpressionNode::Type::Variable, *simplified);
                }
            }
            if (arg->type == ExpressionNode::Type::Number) {
                try {
                    double value = std::get<double>(arg->value);
                    double result;
                    if (name == "sqrt") {
                        if (value < 0) throw std::runtime_error("Square root of negative number");
                        result = std::sqrt(value);
                    } else if (name == "sin") result = std::sin(value);
                    else if (name == "cos") result = std::cos(value);
                    else if (name == "tan") result = std::tan(value);
                    else if (name == "log" || name == "ln") {
                        if (value <= 0) throw std::runtime_error("Logarithm of non-positive number");
                        result = std::log(value);
                    } else if (name == "exp") result = std::exp(value);
                    else if (name == "abs") result = std::abs(value);
                    else if (plugins.count(name)) result = plugins[name]->apply(value);
                    else if (auto res = script_engine.execute(name, value); res) result = *res;
                    else return expr;
                    return std::make_shared<ExpressionNode>(ExpressionNode::Type::Number, result);
                } catch (const std::exception&) {
                    return expr;
                }
            }
        }
        return expr;
    }

    double evaluate_symbolic(const std::shared_ptr<ExpressionNode>& node) {
        if (!node) throw std::runtime_error("Null expression");
        switch (node->type) {
            case ExpressionNode::Type::Number: return std::get<double>(node->value);
            case ExpressionNode::Type::Variable: {
                std::lock_guard<std::mutex> lock(mtx);
                auto var = std::get<std::string>(node->value);
                if (!variables.count(var)) throw std::runtime_error("Undefined variable: " + var);
                return variables[var];
            }
            case ExpressionNode::Type::Operator: {
                if (node->children.size() != 2) throw std::runtime_error("Invalid operator arity");
                char op = std::get<char>(node->value);
                double left = evaluate_symbolic(node->children[0]);
                double right = evaluate_symbolic(node->children[1]);
                switch (op) {
                    case '+': return left + right;
                    case '-': return left - right;
                    case '*': return left * right;
                    case '/': if (std::abs(right) < precision) throw std::runtime_error("Division by zero"); return left / right;
                    case '^': return std::pow(left, right);
                    case '%': if (std::abs(right) < precision) throw std::runtime_error("Modulo by zero"); return std::fmod(left, right);
                    default: throw std::runtime_error("Unknown operator: " + std::string(1, op));
                }
            }
            case ExpressionNode::Type::Function: {
                auto [name, arg] = std::get<std::pair<std::string, std::shared_ptr<ExpressionNode>>>(node->value);
                double val = evaluate_symbolic(arg);
                if (name == "sqrt") {
                    if (val < 0) throw std::runtime_error("Square root of negative number");
                    return std::sqrt(val);
                } else if (name == "sin") return std::sin(val);
                else if (name == "cos") return std::cos(val);
                else if (name == "tan") return std::tan(val);
                else if (name == "log" || name == "ln") {
                    if (val <= 0) throw std::runtime_error("Logarithm of non-positive number");
                    return std::log(val);
                } else if (name == "exp") return std::exp(val);
                else if (name == "abs") return std::abs(val);
                else if (plugins.count(name)) return plugins[name]->apply(val);
                else if (auto res = script_engine.execute(name, val); res) return *res;
                throw std::runtime_error("Unknown function: " + name);
            }
        }
        return 0;
    }

    double parse_factor() {
        skip_whitespace();
        if (current_char == '(') {
            advance();
            double result = parse_expression();
            if (current_char != ')') throw std::runtime_error("Missing ')'");
            advance();
            return result;
        }
        if (std::isdigit(current_char) || current_char == '-' || current_char == '+' || current_char == '.') {
            return parse_number();
        }
        if (std::isalpha(current_char) || current_char == '_') {
            std::string id = parse_identifier();
            skip_whitespace();
            if (current_char == '(') {
                advance();
                double arg = parse_expression();
                if (current_char != ')') throw std::runtime_error("Missing ')' in " + id);
                advance();
                if (id == "sqrt") {
                    if (arg < 0) throw std::runtime_error("Square root of negative number");
                    return std::sqrt(arg);
                } else if (id == "sin") return std::sin(arg);
                else if (id == "cos") return std::cos(arg);
                else if (id == "tan") return std::tan(arg);
                else if (id == "log" || id == "ln") {
                    if (arg <= 0) throw std::runtime_error("Logarithm of non-positive number");
                    return std::log(arg);
                } else if (id == "exp") return std::exp(arg);
                else if (id == "abs") return std::abs(arg);
                else if (plugins.count(id)) return plugins[id]->apply(arg);
                else if (auto res = script_engine.execute(id, arg); res) return *res;
                throw std::runtime_error("Unknown function: " + id);
            }
            std::lock_guard<std::mutex> lock(mtx);
            if (variables.count(id)) return variables[id];
            throw std::runtime_error("Undefined variable: " + id);
        }
        throw std::runtime_error("Invalid factor at position " + std::to_string(pos));
    }

    double parse_power() {
        double result = parse_factor();
        skip_whitespace();
        while (current_char == '^') {
            advance();
            double exp = parse_factor();
            result = std::pow(result, exp);
            skip_whitespace();
        }
        return result;
    }

    double parse_term() {
        double result = parse_power();
        skip_whitespace();
        while (current_char == '*' || current_char == '/' || current_char == '%') {
            char op = current_char;
            advance();
            double next = parse_power();
            if (op == '*') result *= next;
            else if (op == '/') {
                if (std::abs(next) < precision) throw std::runtime_error("Division by zero");
                result /= next;
            } else if (op == '%') {
                if (std::abs(next) < precision) throw std::runtime_error("Modulo by zero");
                result = std::fmod(result, next);
            }
            skip_whitespace();
        }
        return result;
    }

    double parse_expression() {
        double result = parse_term();
        skip_whitespace();
        while (current_char == '+' || current_char == '-') {
            char op = current_char;
            advance();
            double next = parse_term();
            if (op == '+') result += next;
            else result -= next;
            skip_whitespace();
        }
        return result;
    }

    void parse_assignment() {
        std::string var = parse_identifier();
        skip_whitespace();
        if (current_char != '=') throw std::runtime_error("Expected '=' after " + var);
        advance();
        double val = parse_expression();
        {
            std::lock_guard<std::mutex> lock(mtx);
            variables[var] = val;
        }
        logger->log(var + " = " + std::to_string(val));
    }

    double parallel_evaluate(const std::shared_ptr<ExpressionNode>& expr) {
        if (!parallel_mode || expr->type != ExpressionNode::Type::Operator || expr->children.size() != 2) {
            return evaluate_symbolic(expr);
        }
        std::future<double> left_f = std::async(std::launch::async, [this, &expr] { return evaluate_symbolic(expr->children[0]); });
        std::future<double> right_f = std::async(std::launch::async, [this, &expr] { return evaluate_symbolic(expr->children[1]); });
        double left = left_f.get(), right = right_f.get();
        char op = std::get<char>(expr->value);
        switch (op) {
            case '+': return left + right;
            case '-': return left - right;
            case '*': return left * right;
            case '/': if (std::abs(right) < precision) throw std::runtime_error("Division by zero"); return left / right;
            case '^': return std::pow(left, right);
            case '%': if (std::abs(right) < precision) throw std::runtime_error("Modulo by zero"); return std::fmod(left, right);
            default: throw std::runtime_error("Unknown operator");
        }
    }

public:
    SmartMathInterpreter(std::shared_ptr<ILogger> log = std::make_shared<ConsoleLogger>())
        : logger(log), script_engine(log) {
        initialize();
    }

    void set_input(const std::string& new_input) {
        std::lock_guard<std::mutex> lock(mtx);
        input = new_input;
        pos = 0;
        current_char = input.empty() ? '\0' : input[0];
    }

    std::variant<double, std::string> evaluate() {
        std::lock_guard<std::mutex> lock(mtx);
        skip_whitespace();
        if (input.empty() || current_char == '\0') throw std::runtime_error("Empty input");

        if (std::isalpha(current_char) || current_char == '_') {
            size_t start_pos = pos;
            std::string id = parse_identifier();
            skip_whitespace();
            if (current_char == '=') {
                pos = start_pos;
                current_char = input[pos];
                parse_assignment();
                history.emplace_back(input, 0.0, timestamp());
                return 0.0;
            }
            pos = start_pos;
            current_char = input[pos];
        }

        if (symbolic_mode) {
            auto expr = simplify(parse_symbolic_expression());
            skip_whitespace();
            if (current_char != '\0') throw std::runtime_error("Trailing non-whitespace characters after symbolic expression");
            double result = parallel_evaluate(expr);
            history.emplace_back(input, result, timestamp());
            return result;
        }

        double result = parse_expression();
        skip_whitespace();
        if (current_char != '\0') throw std::runtime_error("Trailing non-whitespace characters after expression");
        history.emplace_back(input, result, timestamp());
        return result;
    }

    std::string timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::string ts = std::ctime(&time);
        ts.pop_back();
        return ts;
    }

    void handle_command(const std::string& cmd) {
        std::lock_guard<std::mutex> lock(mtx);
        if (cmd == "history") {
            for (size_t i = 0; i < history.size(); ++i) {
                auto [in, res, ts] = history[i];
                std::visit([&](auto&& val) {
                    std::cout << i + 1 << ": " << in << " = " << val << " [" << ts << "]\n";
                }, res);
            }
        } else if (cmd == "save") {
            std::ofstream out("state.dat");
            if (!out) throw std::runtime_error("Failed to open state.dat for writing");
            for (const auto& [var, val] : variables) out << var << " " << val << "\n";
            logger->log("State saved to state.dat");
        } else if (cmd == "load") {
            std::ifstream in("state.dat");
            if (!in) throw std::runtime_error("Failed to open state.dat for reading");
            std::string var; double val;
            while (in >> var >> val) variables[var] = val;
            logger->log("State loaded from state.dat");
        } else if (cmd == "parallel") {
            parallel_mode = !parallel_mode;
            logger->log("Parallel mode " + std::string(parallel_mode ? "on" : "off"));
        } else if (cmd == "symbolic") {
            symbolic_mode = !symbolic_mode;
            logger->log("Symbolic mode " + std::string(symbolic_mode ? "on" : "off"));
        } else if (cmd == "debug") {
            debug_mode = !debug_mode;
            logger->log("Debug mode " + std::string(debug_mode ? "on" : "off"));
        } else if (cmd.find("script ") == 0) {
            auto parts = split(cmd, ' ');
            if (parts.size() >= 3) {
                std::string code = parts[2];
                for (size_t i = 3; i < parts.size(); ++i) code += " " + parts[i];
                script_engine.define(parts[1], code);
            } else {
                std::cout << "Usage: script <name> <code>\n";
            }
        } else if (cmd == "help") {
            std::cout << "Commands:\n"
                      << "  history - Show calculation history\n"
                      << "  save - Save variables to state.dat\n"
                      << "  load - Load variables from state.dat\n"
                      << "  parallel - Toggle parallel evaluation\n"
                      << "  symbolic - Toggle symbolic mode\n"
                      << "  debug - Toggle debug mode\n"
                      << "  script <name> <code> - Define a script (e.g., 'script double return x * 2')\n"
                      << "  help - Show this help\n"
                      << "  exit - Quit the interpreter\n";
        } else {
            std::cout << "Suggestion: " << suggest_command(cmd) << "\n";
        }
    }

    std::vector<std::string> split(const std::string& s, char delim) {
        std::vector<std::string> result;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) if (!item.empty()) result.push_back(item);
        return result;
    }

    std::string suggest_command(const std::string& partial) {
        std::vector<std::string> cmds = {"history", "save", "load", "parallel", "symbolic", "debug", "script", "help"};
        for (const auto& cmd : cmds) if (cmd.find(partial) == 0) return cmd;
        return "No match";
    }
};

int main() {
    try {
        auto multi_logger = std::make_shared<MultiLogger>();
        multi_logger->addLogger(std::make_unique<ConsoleLogger>());
        try {
            multi_logger->addLogger(std::make_unique<FileLogger>("interpreter.log"));
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not initialize file logger: " << e.what() << "\n";
            std::cerr << "Continuing with console logging only...\n";
        }
        SmartMathInterpreter interp(multi_logger);

        std::cout << "Smart Math Interpreter v5.0.1\n"
                  << "Features: Symbolic algebra, parallel eval, scripting, multi-logging, C++17\n"
                  << "Commands: history, save, load, parallel, symbolic, debug, script <name> <code>, help, exit\n>>> ";

        while (true) {
            std::string input;
            std::getline(std::cin, input);
            if (input.empty()) continue;
            if (input == "exit") break;

            try {
                interp.set_input(input);
                bool is_command = true;
                for (char c : input) {
                    if (!std::isalpha(c) && !std::isspace(c) && c != '_') {
                        is_command = false;
                        break;
                    }
                }
                if (is_command) {
                    interp.handle_command(input);
                } else {
                    auto start = std::chrono::high_resolution_clock::now();
                    auto result = interp.evaluate();
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                    if (std::holds_alternative<double>(result) && input.find('=') == std::string::npos) {
                        std::cout << std::fixed << std::setprecision(6) << std::get<double>(result)
                                  << " (" << duration << " μs)\n";
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
            }
            std::cout << ">>> ";
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}