"""
Optimization algorithm implementations.
"""

from typing import Dict, Any, List, Callable, Optional
import itertools
import random
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GridOptimizer:
    """Grid search optimization"""
    
    def __init__(self):
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf')
        self.all_results: List[Dict[str, Any]] = []
        self.current_trial = 0
        self.total_trials = 0
    
    def optimize(self, evaluate_func: Callable[[Dict[str, Any]], float],
                n_trials: int = None, **kwargs) -> Dict[str, Any]:
        """Run grid search optimization"""
        parameter_space = kwargs.get('parameter_space', {})
        
        if not parameter_space:
            raise ValueError("parameter_space must be provided in kwargs")
        
        # Generate all combinations
        param_combinations = self._generate_combinations(parameter_space)
        self.total_trials = len(param_combinations)
        
        # Limit trials if specified
        if n_trials:
            param_combinations = param_combinations[:n_trials]
            self.total_trials = min(self.total_trials, n_trials)
        
        logger.info(f"Starting grid search with {len(param_combinations)} trials")
        
        # Evaluate each combination
        for params in param_combinations:
            self.current_trial += 1
            
            try:
                score = evaluate_func(params)
                
                # Record result
                result = {
                    'params': params.copy(),
                    'score': score,
                    'trial': self.current_trial,
                    'timestamp': datetime.now()
                }
                self.all_results.append(result)
                
                # Update best if needed
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    logger.info(f"New best score: {score} with params: {params}")
                    
            except Exception as e:
                logger.error(f"Error evaluating {params}: {e}")
                continue
        
        logger.info(f"Grid search complete. Best score: {self.best_score}")
        return self.best_params
    
    def _generate_combinations(self, space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations"""
        keys = list(space.keys())
        values = [space[key] for key in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get best parameters found"""
        return self.best_params.copy() if self.best_params else {}
    
    def get_best_score(self) -> float:
        """Get best score achieved"""
        return self.best_score
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all trials"""
        return self.all_results.copy()


class BayesianOptimizer:
    """Bayesian optimization using surrogate model"""
    
    def __init__(self, acquisition_function: str = 'expected_improvement'):
        self.acquisition_function = acquisition_function
        self.observations: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf')
        self.surrogate_model = None
    
    def optimize(self, evaluate_func: Callable[[Dict[str, Any]], float],
                n_trials: int = 100, **kwargs) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        parameter_space = kwargs.get('parameter_space', {})
        
        if not parameter_space:
            raise ValueError("parameter_space must be provided in kwargs")
        
        # Initial random exploration
        n_initial = min(10, n_trials // 4)
        initial_params = self._random_sample(parameter_space, n_initial)
        
        logger.info(f"Starting Bayesian optimization with {n_initial} initial samples")
        
        # Evaluate initial points
        for params in initial_params:
            score = evaluate_func(params)
            self._update_observations(params, score)
        
        # Bayesian optimization loop
        for i in range(n_trials - n_initial):
            # Fit surrogate model (simplified - would use Gaussian Process in production)
            self._fit_surrogate_model()
            
            # Select next point using acquisition function
            next_params = self._select_next_point(parameter_space)
            
            # Evaluate
            score = evaluate_func(next_params)
            self._update_observations(next_params, score)
            
            logger.debug(f"Trial {n_initial + i + 1}/{n_trials}: score={score}")
        
        logger.info(f"Bayesian optimization complete. Best score: {self.best_score}")
        return self.best_params
    
    def _random_sample(self, space: Dict[str, Any], n_samples: int) -> List[Dict[str, Any]]:
        """Generate random samples from parameter space"""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param_name, param_def in space.items():
                if isinstance(param_def, list):
                    # Discrete values
                    sample[param_name] = random.choice(param_def)
                elif isinstance(param_def, tuple) and len(param_def) == 2:
                    # Continuous range
                    sample[param_name] = random.uniform(param_def[0], param_def[1])
            samples.append(sample)
        
        return samples
    
    def _update_observations(self, params: Dict[str, Any], score: float) -> None:
        """Update observations and best parameters"""
        self.observations.append({
            'params': params.copy(),
            'score': score,
            'timestamp': datetime.now()
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
    
    def _fit_surrogate_model(self) -> None:
        """Fit surrogate model to observations"""
        # Simplified - in production would use sklearn GaussianProcessRegressor
        # For now, just store observations
        pass
    
    def _select_next_point(self, space: Dict[str, Any]) -> Dict[str, Any]:
        """Select next point using acquisition function"""
        # Simplified - would compute acquisition function over space
        # For now, use random with bias towards unexplored regions
        
        # Get one random sample
        candidates = self._random_sample(space, 100)
        
        # Simple heuristic: prefer points far from existing observations
        best_candidate = None
        best_distance = -1
        
        for candidate in candidates:
            min_distance = float('inf')
            for obs in self.observations:
                distance = self._parameter_distance(candidate, obs['params'])
                min_distance = min(min_distance, distance)
            
            if min_distance > best_distance:
                best_distance = min_distance
                best_candidate = candidate
        
        return best_candidate
    
    def _parameter_distance(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate distance between parameter sets"""
        distance = 0
        for key in params1:
            if key in params2:
                val1, val2 = params1[key], params2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    distance += (val1 - val2) ** 2
                elif val1 != val2:
                    distance += 1
        return distance ** 0.5
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get best parameters found"""
        return self.best_params.copy() if self.best_params else {}
    
    def get_best_score(self) -> float:
        """Get best score achieved"""
        return self.best_score
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all trials"""
        return self.observations.copy()


class GeneticOptimizer:
    """Genetic algorithm optimization"""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf')
        self.generation_history: List[Dict[str, Any]] = []
    
    def optimize(self, evaluate_func: Callable[[Dict[str, Any]], float],
                n_trials: int = None, **kwargs) -> Dict[str, Any]:
        """Run genetic algorithm optimization"""
        parameter_space = kwargs.get('parameter_space', {})
        generations = kwargs.get('generations', 20)
        
        if not parameter_space:
            raise ValueError("parameter_space must be provided in kwargs")
        
        logger.info(f"Starting genetic optimization with population={self.population_size}, "
                   f"generations={generations}")
        
        # Initialize population
        population = self._initialize_population(parameter_space)
        
        # Evolution loop
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    score = evaluate_func(individual)
                    fitness_scores.append(score)
                    
                    # Update best
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = individual.copy()
                        
                except Exception as e:
                    fitness_scores.append(float('-inf'))
                    logger.error(f"Error evaluating individual: {e}")
            
            # Record generation stats
            self.generation_history.append({
                'generation': gen,
                'best_score': max(fitness_scores),
                'avg_score': sum(fitness_scores) / len(fitness_scores),
                'timestamp': datetime.now()
            })
            
            logger.debug(f"Generation {gen}: best={max(fitness_scores):.4f}, "
                        f"avg={sum(fitness_scores)/len(fitness_scores):.4f}")
            
            # Select parents and create next generation
            population = self._evolve_population(population, fitness_scores, parameter_space)
        
        logger.info(f"Genetic optimization complete. Best score: {self.best_score}")
        return self.best_params
    
    def _initialize_population(self, space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Initialize random population"""
        population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param_name, param_def in space.items():
                if isinstance(param_def, list):
                    individual[param_name] = random.choice(param_def)
                elif isinstance(param_def, tuple) and len(param_def) == 2:
                    individual[param_name] = random.uniform(param_def[0], param_def[1])
            population.append(individual)
        
        return population
    
    def _evolve_population(self, population: List[Dict[str, Any]], 
                          fitness_scores: List[float],
                          space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create next generation through selection, crossover, and mutation"""
        new_population = []
        
        # Keep best individual (elitism)
        best_idx = fitness_scores.index(max(fitness_scores))
        new_population.append(population[best_idx].copy())
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            # Selection (tournament)
            parent1 = self._tournament_select(population, fitness_scores)
            parent2 = self._tournament_select(population, fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child, space)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_select(self, population: List[Dict[str, Any]], 
                          fitness_scores: List[float],
                          tournament_size: int = 3) -> Dict[str, Any]:
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Uniform crossover between parents"""
        child = {}
        for key in parent1:
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def _mutate(self, individual: Dict[str, Any], space: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate individual parameters"""
        mutated = individual.copy()
        
        # Mutate one random parameter
        param_to_mutate = random.choice(list(space.keys()))
        param_def = space[param_to_mutate]
        
        if isinstance(param_def, list):
            mutated[param_to_mutate] = random.choice(param_def)
        elif isinstance(param_def, tuple) and len(param_def) == 2:
            # Small perturbation for continuous parameters
            current = mutated[param_to_mutate]
            range_size = param_def[1] - param_def[0]
            perturbation = random.gauss(0, range_size * 0.1)
            mutated[param_to_mutate] = max(param_def[0], 
                                           min(param_def[1], current + perturbation))
        
        return mutated
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get best parameters found"""
        return self.best_params.copy() if self.best_params else {}
    
    def get_best_score(self) -> float:
        """Get best score achieved"""
        return self.best_score
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all trials"""
        return self.generation_history.copy()