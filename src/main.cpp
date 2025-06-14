#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>
#include <string>
#include <tuple>
#include <chrono>
#include <unordered_map>

using namespace std;

torch::Device device;

void print_tensor_inline(const std::string& name, const torch::Tensor& t, int precision = 4, int max_elements = 10) {
    //torch::Tensor flat = t.flatten().cpu();
    //std::cout << name << "=tensor([";
    //int64_t size = flat.size(0);
    //std::cout << std::fixed << std::setprecision(precision);
    //for (int64_t i = 0; i < std::min<int64_t>(size, max_elements / 2); ++i) {
    //    std::cout << flat[i].item<double>() << ", ";
    //}
    //if (size > max_elements) {
    //    std::cout << "...";
    //    for (int64_t i = size - max_elements / 2; i < size; ++i) {
    //        std::cout << ", " << flat[i].item<double>();
    //    }
    //}
    //std::cout << "])" << std::endl << std::endl;
}

// Abstract environment interface
class Env {
public:
    struct Space {
        vector<int> shape;
    };

    virtual ~Env() = default;

    // Reset the environment and return initial observation
    virtual pair<vector<float>, unordered_map<string, float>> reset() = 0;

    // Step the environment with given action
    // Returns: observation, reward, terminated flag, truncated flag, extra info
    virtual tuple<vector<float>, float, bool, bool, unordered_map<string, float>> step(const vector<float>& action) = 0;

    virtual void render() = 0;

    virtual Space observation_space() const = 0;

    virtual Space action_space() const = 0;
};

class CartPoleEnv : public Env {
private:
    float gravity = 9.8f;
    float masscart = 1.0f;
    float masspole = 0.1f;
    float total_mass = masspole + masscart;
    float length = 0.5f;  // half the pole's length
    float polemass_length = masspole * length;
    float force_mag = 10.0f;
    float tau = 0.02f;
    float theta_threshold_radians = 12 * 2 * M_PI / 360;
    float x_threshold = 2.4f;

    vector<float> state;
    int steps_beyond_terminated = -1;
    string integrator = "euler";

    mt19937 rng;
    uniform_real_distribution<float> dist{ -0.05f, 0.05f };

public:
    CartPoleEnv() {
        random_device rd;
        rng = mt19937(rd());
    }

    Space observation_space() const override {
        return Space{ {4} };
    }

    Space action_space() const override {
        return Space{ {1} };  // Discrete(2)
    }

    pair<vector<float>, unordered_map<string, float>> reset() override {
        state = { dist(rng), dist(rng), dist(rng), dist(rng) };
        steps_beyond_terminated = -1;
        return { state, {} };
    }

    tuple<vector<float>, float, bool, bool, unordered_map<string, float>> step(const vector<float>& action) override {
        assert(action.size() == 1);

        float x = state[0];
        float x_dot = state[1];
        float theta = state[2];
        float theta_dot = state[3];

        float force = 0.0f;
        if (action[0] > 0)
            force = force_mag;
        else
            force = -force_mag;

        float costheta = cos(theta);
        float sintheta = sin(theta);

        float temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        float thetaacc = (gravity * sintheta - costheta * temp) /
            (length * (4.0f / 3.0f - masspole * costheta * costheta / total_mass));
        float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        if (integrator == "euler") {
            x += tau * x_dot;
            x_dot += tau * xacc;
            theta += tau * theta_dot;
            theta_dot += tau * thetaacc;
        }
        else {
            x_dot += tau * xacc;
            x += tau * x_dot;
            theta_dot += tau * thetaacc;
            theta += tau * theta_dot;
        }

        state = { x, x_dot, theta, theta_dot };

        bool terminated = (x < -x_threshold || x > x_threshold ||
            theta < -theta_threshold_radians || theta > theta_threshold_radians);

        float reward = 0.0f;
        if (!terminated) {
            reward = 1.0f;
        }
        else if (steps_beyond_terminated == -1) {
            steps_beyond_terminated = 0;
            reward = 1.0f;
        }
        else {
            if (steps_beyond_terminated == 0) {
                // warn only once
                printf("Warning: step() called after environment is terminated. Call reset().\n");
            }
            steps_beyond_terminated += 1;
            reward = 0.0f;
        }

        return { state, reward, terminated, false, {} };
    }

    void render() override {
        // Rendering can be implemented here using SDL2, SFML, OpenGL, or skipped.
        printf("State: [%.3f, %.3f, %.3f, %.3f]\n", state[0], state[1], state[2], state[3]);
    }
};

class PendulumEnv : public Env {
public:
    PendulumEnv(const string& render_mode = "", float gravity = 10.0f)
        : g(gravity), render_mode(render_mode) {
        max_speed = 8.0f;
        max_torque = 2.0f;
        dt = 0.05f;
        m = 1.0f;
        l = 1.0f;

        obs_space.shape = { 3 };
        act_space.shape = { 1 };

        random_device rd;
        rng = mt19937(rd());
        dist = uniform_real_distribution<float>(-3.14f, 3.14f);
    }

    pair<vector<float>, unordered_map<string, float>> reset() override {
        float theta = dist(rng);
        float theta_dot = dist(rng) / 4.0f;
        state = { theta, theta_dot };
        last_u = 0.0f;
        return { get_obs(), {} };
    }

    tuple<vector<float>, float, bool, bool, unordered_map<string, float>> step(const vector<float>& action) override {
        float u = clamp(action[0], -max_torque, max_torque);
        last_u = u;

        float theta = state[0];
        float theta_dot = state[1];

        float cost = angle_normalize(theta) * angle_normalize(theta)
            + 0.1f * theta_dot * theta_dot
            + 0.001f * u * u;

        float new_theta_dot = theta_dot + (3.0f * g / (2.0f * l) * sin(theta) + 3.0f / (m * l * l) * u) * dt;
        new_theta_dot = clamp(new_theta_dot, -max_speed, max_speed);
        float new_theta = theta + new_theta_dot * dt;

        state = { new_theta, new_theta_dot };

        return { get_obs(), -cost, false, false, {} };
    }

    void render() override {
        if (render_mode == "human") {
            cout << "Angle: " << state[0] << ", Angular velocity: " << state[1] << ", Torque: " << last_u << "\n";
        }
    }

    Space observation_space() const override {
        return obs_space;
    }

    Space action_space() const override {
        return act_space;
    }

private:
    float g, m, l, dt;
    float max_speed, max_torque;
    float last_u;
    string render_mode;
    vector<float> state;
    Space obs_space, act_space;

    mt19937 rng;
    uniform_real_distribution<float> dist;

    vector<float> get_obs() const {
        float theta = state[0];
        float theta_dot = state[1];
        return { cos(theta), sin(theta), theta_dot };
    }

    float angle_normalize(float x) const {
        return fmodf(x + M_PI, 2.0f * M_PI) - M_PI;
    }

    float clamp(float v, float lo, float hi) const {
        return std::max(lo, std::min(v, hi));
    }
};

class AgentTargetEnv : public Env {
private:
    float x_min = 0.0f, x_max = 10.0f;
    float y_min = 0.0f, y_max = 10.0f;
    float max_step = 0.5f;  // max movement per step in each axis

    vector<float> agent_pos;  // {x, y}
    vector<float> target_pos; // {x, y}

    mt19937 rng;
    uniform_real_distribution<float> dist_x;
    uniform_real_distribution<float> dist_y;

public:
    AgentTargetEnv()
        : dist_x(x_min, x_max), dist_y(y_min, y_max) {
        random_device rd;
        rng = mt19937(rd());
    }

    Space observation_space() const override {
        // Observations: agent_x, agent_y, target_x, target_y
        return Space{ {4} };
    }

    Space action_space() const override {
        // Actions: continuous 2 floats, each in [-1, 1]
        return Space{ {2} };
    }

    pair<vector<float>, unordered_map<string, float>> reset() override {
        agent_pos = { dist_x(rng), dist_y(rng) };
        target_pos = { dist_x(rng), dist_y(rng) };
        return { get_observation(), {} };
    }

    tuple<vector<float>, float, bool, bool, unordered_map<string, float>> step(const vector<float>& action) override {
        // Clip action to [-1,1]
        float dx = max(-1.0f, min(1.0f, action[0])) * max_step;
        float dy = max(-1.0f, min(1.0f, action[1])) * max_step;

        // Update agent position and clip to bounds
        agent_pos[0] = max(x_min, min(x_max, agent_pos[0] + dx));
        agent_pos[1] = max(y_min, min(y_max, agent_pos[1] + dy));

        // Distance to target
        float dist_x = agent_pos[0] - target_pos[0];
        float dist_y = agent_pos[1] - target_pos[1];
        float distance = sqrt(dist_x * dist_x + dist_y * dist_y);

        // Reward and done conditions
        float reward = -0.01f * distance;
        bool done = false;

        if (distance < 1.0f) {
            reward += 1.0f;
            done = true;
        }

        return { get_observation(), reward, done, false, {} };
    }

    void render() override {
        printf("Agent: (%.2f, %.2f), Target: (%.2f, %.2f)\n", agent_pos[0], agent_pos[1], target_pos[0], target_pos[1]);
    }

private:
    vector<float> get_observation() const {
        return { agent_pos[0], agent_pos[1], target_pos[0], target_pos[1] };
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////

class MultivariateNormal {
public:
    torch::Tensor loc;
    torch::Tensor _unbroadcasted_scale_tril;
    torch::Tensor scale_tril_;
    torch::Tensor covariance_matrix_;
    torch::Tensor precision_matrix_;
    std::vector<int64_t> batch_shape;
    std::vector<int64_t> event_shape;

    MultivariateNormal(const torch::Tensor& loc,
        const torch::optional<torch::Tensor>& covariance_matrix = torch::nullopt,
        const torch::optional<torch::Tensor>& precision_matrix = torch::nullopt,
        const torch::optional<torch::Tensor>& scale_tril = torch::nullopt) {
        if (loc.dim() < 1) {
            throw std::invalid_argument("loc must be at least one-dimensional.");
        }

        int specified = (bool)covariance_matrix + (bool)precision_matrix + (bool)scale_tril;
        if (specified != 1) {
            throw std::invalid_argument("Exactly one of covariance_matrix, precision_matrix, or scale_tril must be specified.");
        }

        if (scale_tril.has_value()) {
            if (scale_tril->dim() < 2) {
                throw std::invalid_argument("scale_tril must be at least two-dimensional.");
            }
            auto bshape = broadcast_shapes(scale_tril->sizes().vec(), loc.sizes().vec(), 2, 1);
            scale_tril_ = scale_tril->expand(bshape).contiguous();
            _unbroadcasted_scale_tril = *scale_tril;
            batch_shape = std::vector<int64_t>(bshape.begin(), bshape.end() - 2);
        }
        else if (covariance_matrix.has_value()) {
            if (covariance_matrix->dim() < 2) {
                throw std::invalid_argument("covariance_matrix must be at least two-dimensional.");
            }
            auto bshape = broadcast_shapes(covariance_matrix->sizes().vec(), loc.sizes().vec(), 2, 1);
            covariance_matrix_ = covariance_matrix->expand(bshape).contiguous();
            _unbroadcasted_scale_tril = torch::linalg_cholesky(*covariance_matrix);
            batch_shape = std::vector<int64_t>(bshape.begin(), bshape.end() - 2);
        }
        else {
            if (precision_matrix->dim() < 2) {
                throw std::invalid_argument("precision_matrix must be at least two-dimensional.");
            }
            auto bshape = broadcast_shapes(precision_matrix->sizes().vec(), loc.sizes().vec(), 2, 1);
            precision_matrix_ = precision_matrix->expand(bshape).contiguous();
            _unbroadcasted_scale_tril = torch::linalg_cholesky(torch::linalg_inv(*precision_matrix));
            batch_shape = std::vector<int64_t>(bshape.begin(), bshape.end() - 2);
        }

        auto expanded_shape = batch_shape;
        expanded_shape.push_back(loc.size(-1));
        this->loc = loc.expand(expanded_shape).contiguous();
        event_shape = { loc.size(-1) };
    }

    torch::Tensor scale_tril() const {
        auto shape = batch_shape;
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        return _unbroadcasted_scale_tril.expand(shape);
    }

    torch::Tensor covariance_matrix() const {
        auto L = _unbroadcasted_scale_tril;
        auto shape = batch_shape;
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        return torch::matmul(L, L.transpose(-2, -1)).expand(shape);
    }

    torch::Tensor precision_matrix() const {
        auto shape = batch_shape;
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        return torch::cholesky_inverse(_unbroadcasted_scale_tril).expand(shape);
    }

    torch::Tensor mean() const {
        return loc;
    }

    torch::Tensor mode() const {
        return loc;
    }

    torch::Tensor variance() const {
        auto shape = batch_shape;
        shape.insert(shape.end(), event_shape.begin(), event_shape.end());
        return _unbroadcasted_scale_tril.pow(2).sum(-1).expand(shape);
    }

    torch::Tensor sample(const std::vector<int64_t>& sample_shape = {}) const {
        torch::NoGradGuard no_grad;
        auto shape = sample_shape;
        shape.insert(shape.end(), loc.sizes().begin(), loc.sizes().end());
        auto eps = torch::randn(shape, loc.options());
        auto L = _unbroadcasted_scale_tril;
        auto result = loc + torch::matmul(L, eps.unsqueeze(-1)).squeeze(-1);
        return result;
    }

    torch::Tensor log_prob(const torch::Tensor& value) const {
        auto diff = value - loc;
        auto M = batch_mahalanobis(_unbroadcasted_scale_tril, diff);
        auto half_log_det = _unbroadcasted_scale_tril.diagonal(0, -2, -1).log().sum(-1);
        return -0.5 * (event_shape[0] * std::log(2 * M_PI) + M) - half_log_det;
    }

    torch::Tensor entropy() const {
        auto half_log_det = _unbroadcasted_scale_tril.diagonal(0, -2, -1).log().sum(-1);
        return 0.5 * event_shape[0] * (1.0 + std::log(2 * M_PI)) + half_log_det;
    }

private:
    static torch::Tensor batch_mahalanobis(const torch::Tensor& L, const torch::Tensor& diff) {
        auto solve = torch::linalg_solve_triangular(L, diff.unsqueeze(-1), /*upper=*/false).squeeze(-1);
        return solve.pow(2).sum(-1);
    }

    static std::vector<int64_t> broadcast_shapes(std::vector<int64_t> a, std::vector<int64_t> b, int a_end = 0, int b_end = 0) {
        auto a_prefix = std::vector<int64_t>(a.begin(), a.end() - a_end);
        auto b_prefix = std::vector<int64_t>(b.begin(), b.end() - b_end);
        size_t ndim = std::max(a_prefix.size(), b_prefix.size());
        std::vector<int64_t> result(ndim, 1);
        for (int i = ndim - 1, ai = a_prefix.size() - 1, bi = b_prefix.size() - 1; i >= 0; --i, --ai, --bi) {
            int64_t a_dim = ai >= 0 ? a_prefix[ai] : 1;
            int64_t b_dim = bi >= 0 ? b_prefix[bi] : 1;
            if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                throw std::invalid_argument("Incompatible shapes for broadcasting");
            }
            result[i] = std::max(a_dim, b_dim);
        }
        result.insert(result.end(), a.end() - a_end, a.end());
        return result;
    }
};

struct FeedForwardNNImpl : torch::nn::Module {
    torch::nn::Linear layer1{ nullptr }, layer2{ nullptr }, layer3{ nullptr };

    FeedForwardNNImpl(int in_dim, int out_dim) {
        layer1 = register_module("layer1", torch::nn::Linear(in_dim, 64));
        layer2 = register_module("layer2", torch::nn::Linear(64, 64));
        layer3 = register_module("layer3", torch::nn::Linear(64, out_dim));
    }

    torch::Tensor forward(torch::Tensor obs) {
        try {
            auto activation1 = torch::relu(layer1(obs));
            auto activation2 = torch::relu(layer2(activation1));
            return layer3(activation2);
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in forward: " << e.what() << std::endl;
            throw;  // rethrow or handle as needed
        }
    }
};
TORCH_MODULE(FeedForwardNN);

class PPO {
public:
    FeedForwardNN actor = nullptr;
    FeedForwardNN critic = nullptr;

    PPO(Env& env, const unordered_map<string, float>& hyperparameters)
        : env(env) {

        _init_hyperparameters(hyperparameters);

        obs_dim = env.observation_space().shape[0];
        act_dim = env.action_space().shape[0];

        actor = FeedForwardNN(obs_dim, act_dim);
        critic = FeedForwardNN(obs_dim, 1);

        actor_optim = std::make_unique<torch::optim::Adam>(actor->parameters(), torch::optim::AdamOptions(lr));
        critic_optim = std::make_unique<torch::optim::Adam>(critic->parameters(), torch::optim::AdamOptions(lr));

        cov_var = torch::full({ act_dim }, 0.5);
        cov_mat = cov_var.diag();

        logger["delta_t"] = chrono::high_resolution_clock::now().time_since_epoch().count();
        logger["t_so_far"] = 0;
        logger["i_so_far"] = 0;
    }

    void learn(int total_timesteps) {
        cout << "Learning... Running " << max_timesteps_per_episode
            << " timesteps per episode, " << timesteps_per_batch
            << " timesteps per batch for a total of "
            << total_timesteps << " timesteps" << endl;

        int t_so_far = 0;
        int i_so_far = 0;

        while (t_so_far < total_timesteps) {
            // ALG STEP 2-3: Rollout to collect a batch of trajectories
            auto [batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths] = rollout();

            // Count how many timesteps we collected
            t_so_far += batch_lengths.sum().item<int>();

            // Count iterations
            i_so_far += 1;

            // Logging
            logger["t_so_far"] = t_so_far;
            logger["i_so_far"] = i_so_far;

            // Evaluate current value function and policy log probs
            auto [V, _] = evaluate(batch_obs, batch_acts);

            // Advantage estimation
            torch::Tensor A_k = batch_rtgs - V.detach();

            // Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10);

            ///TEMP
            print_tensor_inline("batch_obs", batch_obs);
            print_tensor_inline("batch_acts", batch_acts);
            print_tensor_inline("batch_log_probs", batch_log_probs);
            print_tensor_inline("batch_rtgs", batch_rtgs);
            print_tensor_inline("batch_lengths", batch_lengths);
            print_tensor_inline("V", V);
            print_tensor_inline("A_k", A_k);

            // PPO update for multiple epochs
            for (int epoch = 0; epoch < n_updates_per_iteration; ++epoch) {
                auto [V, curr_log_probs] = evaluate(batch_obs, batch_acts);

                // Compute ratio of new and old policy probabilities
                torch::Tensor ratios = torch::exp(curr_log_probs - batch_log_probs);

                // Compute surrogate losses
                torch::Tensor surr1 = ratios * A_k;
                torch::Tensor surr2 = torch::clamp(ratios, 1 - clip, 1 + clip) * A_k;

                // Compute losses
                torch::Tensor actor_loss = -torch::min(surr1, surr2).mean();
                torch::Tensor critic_loss = torch::mse_loss(V, batch_rtgs);

                // Backpropagate actor loss
                actor_optim->zero_grad();
                actor_loss.backward({}, /* retain_graph */ true);
                actor_optim->step();

                // Backpropagate critic loss
                critic_optim->zero_grad();
                critic_loss.backward();
                critic_optim->step();

                // Logging actor loss
                logger["actor_loss"] = actor_loss;

                ///TEMP
                print_tensor_inline("V", V);
                print_tensor_inline("curr_log_probs", curr_log_probs);
                print_tensor_inline("ratios", ratios);
                print_tensor_inline("surr1", surr1);
                print_tensor_inline("surr2", surr2);
                print_tensor_inline("actor_loss", actor_loss);
                print_tensor_inline("critic_loss", critic_loss);
            }

            // Print training summary
            _log_summary();

            // Save model
            if (i_so_far % save_freq == 0) {
                torch::save(actor, "./ppo_actor.pt");
                torch::save(critic, "./ppo_critic.pt");
            }
        }
    }

    tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rollout() {
        // Batch data. For more details, check function header.
        vector<torch::Tensor> batch_obs_vec;
        vector<torch::Tensor> batch_acts_vec;
        vector<torch::Tensor> batch_log_probs_vec;
        vector<vector<float>> batch_rewards;
        vector<int> batch_lengths_vec;

        // Episodic data. Keeps track of rewards per episode, will get cleared
        // upon each new episode
        vector<float> ep_rews;

        int t = 0; // Keeps track of how many timesteps we've run so far this batch

        // Keep simulating until we've run more than or equal to specified timesteps per batch
        while (t < timesteps_per_batch) {
            ep_rews.clear(); // rewards collected per episode

            // Reset the environment. Note that obs is short for observation.
            auto [obs, _] = env.reset();
            bool done = false;

            // Run an episode for a maximum of max_timesteps_per_episode timesteps
            for (int ep_t = 0; ep_t < max_timesteps_per_episode; ++ep_t) {
                // If render is specified, render the environment
                if (render && (std::get<int>(logger["i_so_far"]) % render_every_i == 0) && batch_lengths_vec.empty()) {
                    env.render();
                }

                t += 1; // Increment timesteps ran this batch so far

                // Track observations in this batch

                torch::Tensor obs_tensor = torch::from_blob((void*)obs.data(), { (int)obs.size() }, torch::kFloat).clone();
                batch_obs_vec.push_back(obs_tensor);

                // Calculate action and make a step in the env.
                // Note that rew is short for reward.
                auto [action, log_prob] = get_action(obs_tensor);
                torch::Tensor action_tensor = torch::from_blob((void*)action.data(), { (int)action.size() }, torch::kFloat).clone();
                auto [next_obs, rew, terminated, truncated, __] = env.step(action);
                print_tensor_inline("log_prob", log_prob);

                // Don't really care about the difference between terminated or truncated in this, so just combine them
                done = terminated || truncated;

                // Track recent reward, action, and action log probability
                ep_rews.push_back(rew);
                batch_acts_vec.push_back(action_tensor);
                batch_log_probs_vec.push_back(log_prob);

                obs = next_obs;

                // If the environment tells us the episode is terminated, break
                if (done) {
                    break;
                }
            }

            // Track episodic lengths and rewards
            batch_lengths_vec.push_back(ep_rews.size());
            batch_rewards.push_back(ep_rews);
        }

        // Reshape data as tensors in the shape specified in function description, before returning
        torch::Tensor batch_obs = torch::stack(batch_obs_vec).to(torch::kFloat);
        torch::Tensor batch_acts = torch::stack(batch_acts_vec).to(torch::kFloat);
        torch::Tensor batch_log_probs = torch::stack(batch_log_probs_vec).to(torch::kFloat);
        torch::Tensor batch_rtgs = compute_rtgs(batch_rewards); // ALG STEP 4
        torch::Tensor batch_lengths = torch::tensor(batch_lengths_vec, torch::kInt64);

        print_tensor_inline("batch_obs", batch_obs);
        print_tensor_inline("batch_acts", batch_acts);
        print_tensor_inline("batch_log_probs", batch_log_probs);
        print_tensor_inline("batch_rtgs", batch_rtgs);
        print_tensor_inline("batch_lengths", batch_lengths);

        // Log the episodic returns and episodic lengths in this batch.
        logger["batch_rewards"] = batch_rewards;
        logger["batch_lengths"] = batch_lengths_vec;

        return { batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths };
    }

    torch::Tensor compute_rtgs(const vector<vector<float>>& batch_rewards) {
        /**
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rewards - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        */

        // The rewards-to-go (rtg) per episode per batch to return.
        // The shape will be (num timesteps per episode)
        vector<float> batch_rtgs;

        // Iterate through each episode
        for (auto it = batch_rewards.rbegin(); it != batch_rewards.rend(); ++it) {
            const auto& ep_rews = *it;
            float discounted_reward = 0; // The discounted reward so far

            // Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            // discounted return (think about why it would be harder starting from the beginning)
            for (auto rit = ep_rews.rbegin(); rit != ep_rews.rend(); ++rit) {
                discounted_reward = *rit + discounted_reward * gamma;
                batch_rtgs.insert(batch_rtgs.begin(), discounted_reward);
            }
        }

        // Convert the rewards-to-go into a tensor
        return torch::tensor(batch_rtgs, torch::kFloat);
    }

    pair<vector<float>, torch::Tensor> get_action(const torch::Tensor& obs_tensor) {
        // Query the actor network for a mean action
        torch::Tensor mean = actor->forward(obs_tensor);

        // Create a distribution with the mean action and std from the covariance matrix
        auto dist = MultivariateNormal(mean, cov_mat);

        // Sample an action from the distribution
        torch::Tensor action_tensor = dist.sample();

        // Calculate the log probability for that action
        torch::Tensor log_prob_tensor = dist.log_prob(action_tensor);

        print_tensor_inline("obs_tensor", obs_tensor);
        print_tensor_inline("mean", mean);
        print_tensor_inline("action_tensor", action_tensor);
        print_tensor_inline("log_prob_tensor", log_prob_tensor);

        // Detach before returning
        vector<float> action(action_tensor.detach().data_ptr<float>(), action_tensor.detach().data_ptr<float>() + action_tensor.numel());

        return { action, log_prob_tensor.detach() };
    }

    pair<torch::Tensor, torch::Tensor> evaluate(const torch::Tensor& batch_obs, const torch::Tensor& batch_acts) {
        /**
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)

            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        */

        // Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        torch::Tensor V = critic->forward(batch_obs).squeeze();

        // Calculate the log probabilities of batch actions using most recent actor network.
        // This segment of code is similar to that in get_action()
        torch::Tensor mean = actor->forward(batch_obs);
        MultivariateNormal dist(mean, cov_mat);
        torch::Tensor log_probs = dist.log_prob(batch_acts);

        // Return the value vector V of each observation in the batch
        // and log probabilities log_probs of each action in the batch
        return { V, log_probs };
    }

private:
    void _init_hyperparameters(const unordered_map<string, float>& hyperparameters) {
        /**
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                  hyperparameters defined below with custom values.

            Return:
                None
        */

        // Initialize default values for hyperparameters
        // Algorithm hyperparameters
        timesteps_per_batch = 4800;               // Number of timesteps to run per batch
        max_timesteps_per_episode = 1600;         // Max number of timesteps per episode
        n_updates_per_iteration = 5;              // Number of times to update actor/critic per iteration
        lr = 0.005;                              // Learning rate of actor optimizer
        gamma = 0.95;                            // Discount factor to be applied when calculating Rewards-To-Go
        clip = 0.2;                              // Recommended 0.2, helps define the threshold to clip the ratio during SGA

        // Miscellaneous parameters
        render = false;                          // If we should render during rollout
        render_every_i = 10;                    // Only render every n iterations
        save_freq = 10;                         // How often we save in number of iterations
        seed = nullopt;                    // Sets the seed of our program, used for reproducibility of results

        // Change any default values to custom values for specified hyperparameters
        for (const auto& [param, val] : hyperparameters) {
            if (param == "timesteps_per_batch") timesteps_per_batch = static_cast<int>(val);
            else if (param == "max_timesteps_per_episode") max_timesteps_per_episode = static_cast<int>(val);
            else if (param == "n_updates_per_iteration") n_updates_per_iteration = static_cast<int>(val);
            else if (param == "lr") lr = val;
            else if (param == "gamma") gamma = val;
            else if (param == "clip") clip = val;
            else if (param == "render") render = (val != 0.0);
            else if (param == "render_every_i") render_every_i = static_cast<int>(val);
            else if (param == "save_freq") save_freq = static_cast<int>(val);
            else if (param == "seed") seed = static_cast<int>(val);
            // Add more parameters here if needed
        }

        // Sets the seed if specified
        if (seed.has_value()) {
            // Set the seed
            torch::manual_seed(seed.value());
            cout << "Successfully set seed to " << seed.value() << endl;
        }
    }

    void _log_summary() {
        try {
            long long prev_delta_t = std::get<long long>(logger["delta_t"]);
            logger["delta_t"] = chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now().time_since_epoch()).count();
            float delta_t_sec = (std::get<long long>(logger["delta_t"]) - prev_delta_t) / 1e9;

            stringstream delta_t_ss;
            delta_t_ss << fixed << setprecision(2) << delta_t_sec;
            string delta_t = delta_t_ss.str();

            int t_so_far = std::get<int>(logger["t_so_far"]);
            int i_so_far = std::get<int>(logger["i_so_far"]);

            vector<int> batch_lengths = std::get<vector<int>>(logger["batch_lengths"]);
            float avg_ep_lens = 0.0;
            if (!batch_lengths.empty()) {
                avg_ep_lens = accumulate(batch_lengths.begin(), batch_lengths.end(), 0.0) / batch_lengths.size();
            }

            vector<vector<float>> batch_rewards = std::get<vector<vector<float>>>(logger["batch_rewards"]);
            float avg_ep_rews = 0.0;
            if (!batch_rewards.empty()) {
                float sum_rews = 0.0;
                int count = 0;
                for (const auto& ep_rews : batch_rewards) {
                    sum_rews += accumulate(ep_rews.begin(), ep_rews.end(), 0.0);
                    count++;
                }
                avg_ep_rews = sum_rews / count;
            }

            auto actor_loss = std::get<torch::Tensor>(logger["actor_loss"]);
            float avg_actor_loss = 0.0;
            if (actor_loss.numel() > 0) {
                if (actor_loss.dim() == 0) {
                    // scalar tensor
                    avg_actor_loss = actor_loss.item<float>();
                }
                else {
                    float sum_loss = 0.0;
                    for (int i = 0; i < actor_loss.size(0); ++i) {
                        sum_loss += actor_loss[i].item<float>();
                    }
                    avg_actor_loss = sum_loss / actor_loss.size(0);
                }
            }

            stringstream avg_ep_lens_ss, avg_ep_rews_ss, avg_actor_loss_ss;
            avg_ep_lens_ss << fixed << setprecision(2) << avg_ep_lens;
            avg_ep_rews_ss << fixed << setprecision(2) << avg_ep_rews;
            avg_actor_loss_ss << fixed << setprecision(5) << avg_actor_loss;

            cout << endl;
            cout << "-------------------- Iteration #" << i_so_far << " --------------------" << endl;
            cout << "Average Episodic Length: " << avg_ep_lens_ss.str() << endl;
            cout << "Average Episodic Return: " << avg_ep_rews_ss.str() << endl;
            cout << "Average Loss: " << avg_actor_loss_ss.str() << endl;
            cout << "Timesteps So Far: " << t_so_far << endl;
            cout << "Iteration took: " << delta_t << " secs" << endl;
            cout << "------------------------------------------------------" << endl;
            cout << endl;

            logger["batch_lengths"] = vector<int>{};
            logger["batch_rewards"] = vector<vector<float>>{};
            logger["actor_loss"] = torch::Tensor();
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in _log_summary: " << e.what() << std::endl;
        }
    }

    Env& env;
    int obs_dim;
    int act_dim;
    float lr;

    torch::Tensor cov_var;
    torch::Tensor cov_mat;


    unique_ptr<torch::optim::Adam> actor_optim;
    unique_ptr<torch::optim::Adam> critic_optim;

    using LoggerValue = variant<
        string,
        float,
        int,
        vector<float>,
        vector<int>,
        vector<torch::Tensor>,
        vector<vector<float>>,
        long long,
        torch::Tensor
    >;

    unordered_map<string, LoggerValue> logger;
    vector<int> batch_lengths;
    vector<float> batch_rewards;

    // Algorithm hyperparameters
    int timesteps_per_batch;
    int max_timesteps_per_episode;
    int n_updates_per_iteration;
    float gamma;
    float clip;

    // Miscellaneous parameters
    bool render;
    int render_every_i;
    int save_freq;
    optional<int> seed;
};


void train(
    Env& env,
    const unordered_map<string, float>& hyperparameters,
    const string& actor_model,
    const string& critic_model
) {
    /**
     * Trains the model.
     *
     * Parameters:
     *   env - the environment to train on
     *   hyperparameters - map of hyperparameters
     *   actor_model - path to actor model file if continuing training
     *   critic_model - path to critic model file if continuing training
     *
     * Return:
     *   None
     */
    cout << "Training" << endl;

    PPO model(env, hyperparameters);  // Construct PPO with environment and hyperparameters

    if (!actor_model.empty() && !critic_model.empty()) {
        cout << "Loading in " << actor_model << " and " << critic_model << "..." << endl;
        torch::load(model.actor, actor_model);
        torch::load(model.critic, critic_model);
        cout << "Successfully loaded." << endl;
    }
    else if (!actor_model.empty() || !critic_model.empty()) {
        cerr << "Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!" << endl;
        exit(0);
    }
    else {
        cout << "Training from scratch." << endl;
    }

    // Train PPO model for a large number of timesteps
    model.learn(200000000);
}

void test(Env& env, const string& actor_model) {
    /**
     * Tests the model.
     *
     * Parameters:
     *   env - the environment to test on
     *   actor_model - path to actor model file
     *
     * Return:
     *   None
     */
    cout << "Testing " << actor_model << endl;

    if (actor_model.empty()) {
        cerr << "Didn't specify model file. Exiting." << endl;
        exit(0);
    }

    int obs_dim = env.observation_space().shape[0];
    int act_dim = env.action_space().shape[0];

    FeedForwardNN policy(obs_dim, act_dim);

    torch::load(policy, actor_model);

    //eval_policy(policy, env, /*render=*/true);
}

int main(int argc, char* argv[]) {
    /**
     * Main function to run.
     *
     * Parameters:
     *   argc, argv - command line arguments
     *
     * Return:
     *   None
     */

     // Parse arguments (you need to implement get_args or use your favorite CLI parser)
     // 
    // Hyperparameters for PPO (can be customized here)
    unordered_map<string, float> hyperparameters = {
        {"timesteps_per_batch", 2048},
        {"max_timesteps_per_episode", 200},
        {"gamma", 0.99},
        {"n_updates_per_iteration", 10},
        {"lr", 3e-4},
        {"clip", 0.2},
        {"render", 0},           // Using 1 for true
        {"render_every_i", 10}
    };


    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available. GPU will be used.\n";
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);

        if (err != cudaSuccess || deviceCount == 0) {
            std::cout << "Failed to get CUDA device count or no devices found.\n";
            return 1;
        }

        std::cout << "CUDA device count: " << deviceCount << "\n";

        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, 0);
        if (err == cudaSuccess) {
            std::cout << "Current device name: " << deviceProp.name << "\n";
        }
        else {
            std::cout << "Failed to get device properties.\n";
        }

        device = torch::Device(torch::kCUDA, 0);
    }
    else {
        std::cout << "CUDA is NOT available. CPU will be used.\n";
        device = torch::Device(torch::kCPU);
    }


    try {
        CartPoleEnv env;
        if (true) {
            train(env, hyperparameters, "", "");
        }
        else {
            test(env, ""); // only load the actor model
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred." << std::endl;
    }

    return 0;
}
