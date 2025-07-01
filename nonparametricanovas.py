import pandas as pd
import numpy as np
from scipy import stats
import warnings
import pingouin as pg  # Required for some ANOVA calculations
from itertools import combinations
from statsmodels.stats.multitest import multipletests  # For better multiple comparison adjustments

class NonParametricANOVA:
    """Base class for non-parametric ANOVA alternatives using rank transformation + permutation tests."""
    
    def __init__(self, n_perm=1000, random_state=None, alpha=0.05):
        """
        Initialize the non-parametric ANOVA test.
        
        Parameters:
        -----------
        n_perm : int
            Number of permutations to run
        random_state : int or None
            Seed for random number generator
        alpha : float
            Significance level
        """
        # Remove the problematic line: super().__init__(n_perm, random_state, alpha)
        self.test_name = "Non-parametric Mixed ANOVA (Rank + Permutation)"
        self.n_perm = n_perm
        self.rng = np.random.default_rng(random_state)
        self.alpha = alpha
        self.results = {}
        
    def _bootstrap_ci(self, data1, data2, stat_func, n_bootstrap=1000, alpha=0.05):
        """Bootstrap confidence interval for a statistic between two samples."""
        boot_stats = []
        n1, n2 = len(data1), len(data2)
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(data1, n1, replace=True)
            sample2 = np.random.choice(data2, n2, replace=True)
            boot_stats.append(stat_func(sample1, sample2))
        lower = np.percentile(boot_stats, 100 * alpha / 2)
        upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        return lower, upper

    def _rank_biserial_effect_size(self, data1, data2):
        """Rank-biserial correlation for two independent samples."""
        u, _ = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        n1, n2 = len(data1), len(data2)
        rbc = 1 - (2 * u) / (n1 * n2)
        return rbc

    def _standardize_results(self, source_data):
        """Create standardized result structure to match parametric ANOVA outputs, now with posthoc, effect size, and CI."""
        std_results = {
            "test_type": "non-parametric",
            "test_name": self.test_name,
            "statistic_type": "F",  # Using F as the test statistic for compatibility
            "error": None,
            "effects": []
        }
        for effect_name, effect_data in source_data.items():
            effect = {
                "name": effect_name,
                "F": effect_data.get("F", None),
                "p": effect_data.get("p", None),
                "significant": effect_data.get("p", 1.0) < self.alpha,
                "df_num": effect_data.get("df_num", None),
                "df_den": effect_data.get("df_den", None),
                "permutation_test": True,
                "effect_size": effect_data.get("effect_size", None),
                "ci_lower": effect_data.get("ci_lower", None),
                "ci_upper": effect_data.get("ci_upper", None),
                "posthoc_tests": effect_data.get("posthoc_tests", None)
            }
            std_results["effects"].append(effect)
        return std_results
    
    def _posthoc_mannwhitney(self, df, dv, factor):
        """Pairwise Mann-Whitney U tests with Holm-Sidak correction for independent groups."""
        levels = df[factor].unique()
        pairs = list(combinations(levels, 2))
        results = []
        pvals = []
        
        for a, b in pairs:
            data1 = df.loc[df[factor]==a, dv]
            data2 = df.loc[df[factor]==b, dv]
            stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            pvals.append(p)
            results.append({"level1": a, "level2": b, "statistic": stat, "p_raw": p})
            
        # Apply Holm-Sidak correction
        valid_pvals = np.array([p for p in pvals if p is not None and not np.isnan(p)])
        if len(valid_pvals) > 0:
            valid_indices = [i for i, p in enumerate(pvals) if p is not None and not np.isnan(p)]
            reject, p_adj = multipletests(valid_pvals, alpha=self.alpha, method='holm-sidak')[:2]
            
            # Set adjusted p-values and significance flags
            for i, r in enumerate(results):
                if i in valid_indices:
                    idx = valid_indices.index(i)
                    r["p_adj"] = p_adj[idx]
                    r["significant"] = reject[idx]
                else:
                    r["p_adj"] = 1.0
                    r["significant"] = False
                r["adjustment"] = "holm-sidak"
        
        return results
    def _freedman_lane_permutation(self, df, dv, between, effect, n_perm, model_type, subject=None, within=None):
        """
        Perform Freedman–Lane permutation for a given effect on rank-transformed data.
        """
        print(f"DEBUG: Starting Freedman-Lane for effect='{effect}', model='{model_type}'")
        print(f"DEBUG: Parameters - between={between}, within={within}, subject={subject}")
        # 1. Standardize effect name for Pingouin (interaction: 'A * B')
        def _pg_effect_name(eff, between, within):
            if isinstance(eff, str) and '*' in eff:
                return eff.replace(':', ' * ')
            if eff in between or (within and eff in within):
                return eff
            return eff
        effect_pg = _pg_effect_name(effect, between, within)
        # 2. Fit reduced model (exclude effect)
        if model_type == 'anova':
            reduced_between = [b for b in between if b != effect_pg and b != effect_pg.replace(' * ', '')]
            if reduced_between:
                group_means = df.groupby(reduced_between)['rank'].transform('mean')
            else:
                group_means = np.repeat(df['rank'].mean(), len(df))
            fitted = group_means
        elif model_type == 'rm_anova':
            fitted = df.groupby(subject)['rank'].transform('mean')
        elif model_type == 'mixed_anova':
            reduced_between = [b for b in between if b != effect_pg and b != effect_pg.replace(' * ', '')]
            reduced_within = [w for w in within if w != effect_pg and w != effect_pg.replace(' * ', '')]
            if reduced_between or reduced_within:
                group_cols = []
                if reduced_between:
                    group_cols.append(reduced_between[0])
                if reduced_within:
                    group_cols.append(reduced_within[0])
                if group_cols:
                    fitted = df.groupby([subject] + group_cols)['rank'].transform('mean')
                else:
                    fitted = df.groupby(subject)['rank'].transform('mean')
            else:
                fitted = df.groupby(subject)['rank'].transform('mean')
        else:
            raise ValueError('Unknown model_type for Freedman–Lane')
        resid = df['rank'] - fitted
        obs_df = df.copy()
        obs_df['rank_FL'] = fitted + resid
        # 3. Observed F - after calculating obs_F, add:
        if model_type == 'anova':
            obs_aov = pg.anova(data=obs_df, dv='rank_FL', between=between, detailed=True)
            obs_F = float(obs_aov.loc[obs_aov['Source'] == effect_pg, 'F'])
        elif model_type == 'rm_anova':
            obs_aov = pg.rm_anova(data=obs_df, dv='rank_FL', within=within[0], subject=subject, detailed=True)
            obs_F = float(obs_aov.loc[obs_aov['Source'] == effect_pg, 'F'])
        elif model_type == 'mixed_anova':
            obs_aov = pg.mixed_anova(data=obs_df, dv='rank_FL', between=between[0], within=within[0], subject=subject, detailed=True)
            obs_F = float(obs_aov.loc[obs_aov['Source'] == effect_pg, 'F'])
        
        # Add this check to handle inf/nan F values
        if not np.isfinite(obs_F):
            print(f"Warning: Observed F-statistic is not finite (inf or NaN). Setting to a large finite value.")
            obs_F = 1000.0  # Use a large value instead of infinity
        # 4. Permutations
        perm_F = np.empty(n_perm)
        if model_type == 'rm_anova' and subject is not None:
            # Permute residuals within subjects
            for i in range(n_perm):
                perm_resid = resid.groupby(df[subject]).transform(lambda x: self.rng.permutation(x))
                df_perm = df.copy()
                df_perm['rank_FL'] = fitted + perm_resid
                try:
                    perm_aov = pg.rm_anova(data=df_perm, dv='rank_FL', within=within[0], subject=subject, detailed=False)
                    perm_F[i] = float(perm_aov.loc[perm_aov['Source'] == effect_pg, 'F'])
                except:
                    perm_F[i] = 0
        else:
            for i in range(n_perm):
                perm_resid = self.rng.permutation(resid)
                df_perm = df.copy()
                df_perm['rank_FL'] = fitted + perm_resid
                try:
                    if model_type == 'anova':
                        perm_aov = pg.anova(data=df_perm, dv='rank_FL', between=between, detailed=False)
                        # FIX: Use square brackets for loc, not parentheses
                        perm_F[i] = float(perm_aov.loc[perm_aov['Source'] == effect_pg, 'F'])
                    elif model_type == 'mixed_anova':
                        perm_aov = pg.mixed_anova(data=df_perm, dv='rank_FL', between=between[0], within=within[0], subject=subject)
                        # FIX: Use square brackets for loc, not parentheses
                        perm_F[i] = float(perm_aov.loc[perm_aov['Source'] == effect_pg, 'F'])
                    elif model_type == 'rm_anova':  # Add missing rm_anova case
                        perm_aov = pg.rm_anova(data=df_perm, dv='rank_FL', within=within[0], subject=subject, detailed=False)
                        perm_F[i] = float(perm_aov.loc[perm_aov['Source'] == effect_pg, 'F'])
                except Exception as perm_e:  # Properly capture the exception
                    print(f"Warning in permutation {i}: {str(perm_e)}")
                    perm_F[i] = 0
        print(f"DEBUG: Freedman-Lane complete. obs_F={obs_F:.4f}, {len(perm_F)} permutations.")
        return obs_F, perm_F
        
    

class NonParametricTwoWayANOVA(NonParametricANOVA):
    """Non-parametric Two-Way ANOVA using rank transformation + permutation test."""
    
    def __init__(self, n_perm=1000, random_state=None, alpha=0.05):
        super().__init__(n_perm, random_state, alpha)
        self.test_name = "Non-parametric Two-Way ANOVA (Rank + Permutation)"
    
    def run(self, df, dv, factors):
        """
        Run non-parametric Two-Way ANOVA using Freedman–Lane permutation for each effect.
        """
        try:
            factorA, factorB = factors[0], factors[1]
            
            # Store original data before aggregation for Excel export
            original_df = df.copy()
            
            # ======= AGGREGATE TRIPLICATES/REPLICATES =======
            print(f"DEBUG: Original data shape: {df.shape}")
            counts = df.groupby([factorA, factorB])[dv].count().reset_index(name="n_obs")
            max_obs = counts["n_obs"].max()
            print(f"DEBUG: Max observations per cell: {max_obs}")
            
            if max_obs > 1:
                print(f"DEBUG: Found replicates - aggregating by mean...")
                df = df.groupby([factorA, factorB], as_index=False)[dv].mean()
                print(f"DEBUG: After aggregation shape: {df.shape}")
            else:
                print("DEBUG: No replicates found, proceeding with original data")
            # ================================================
            
            interaction = f"{factorA} * {factorB}"
            df['rank'] = df[dv].rank(method='average')
            observed_aov = pg.anova(data=df, dv='rank', between=[factorA, factorB], detailed=True)
            f_obs_A = float(observed_aov.loc[observed_aov['Source']==factorA, 'F'])
            f_obs_B = float(observed_aov.loc[observed_aov['Source']==factorB, 'F'])
            f_obs_interaction = float(observed_aov.loc[observed_aov['Source']==interaction, 'F'])
            df_num_A = int(observed_aov.loc[observed_aov['Source']==factorA, 'DF'])
            df_den_A = int(observed_aov.loc[observed_aov['Source']=='Residual', 'DF'])
            df_num_B = int(observed_aov.loc[observed_aov['Source']==factorB, 'DF'])
            df_den_B = int(observed_aov.loc[observed_aov['Source']=='Residual', 'DF'])
            df_num_interaction = int(observed_aov.loc[observed_aov['Source']==interaction, 'DF'])
            df_den_interaction = int(observed_aov.loc[observed_aov['Source']=='Residual', 'DF'])
            f_obs_A, f_perms_A = self._freedman_lane_permutation(df, dv, [factorA, factorB], factorA, self.n_perm, 'anova')
            f_obs_B, f_perms_B = self._freedman_lane_permutation(df, dv, [factorA, factorB], factorB, self.n_perm, 'anova')
            f_obs_interaction, f_perms_interaction = self._freedman_lane_permutation(df, dv, [factorA, factorB], interaction, self.n_perm, 'anova')
            p_A = (1 + np.sum(f_perms_A >= f_obs_A)) / (1 + self.n_perm)
            p_B = (1 + np.sum(f_perms_B >= f_obs_B)) / (1 + self.n_perm)
            p_interaction = (1 + np.sum(f_perms_interaction >= f_obs_interaction)) / (1 + self.n_perm)
            effect_size_A, ci_A = None, (None, None)
            effect_size_B, ci_B = None, (None, None)
            if df[factorA].nunique() == 2:
                levels = df[factorA].unique()
                data1 = df.loc[df[factorA]==levels[0], 'rank']
                data2 = df.loc[df[factorA]==levels[1], 'rank']
                effect_size_A = self._rank_biserial_effect_size(data1, data2)
                ci_A = self._bootstrap_ci(data1, data2, self._rank_biserial_effect_size)
            if df[factorB].nunique() == 2:
                levels = df[factorB].unique()
                data1 = df.loc[df[factorB]==levels[0], 'rank']
                data2 = df.loc[df[factorB]==levels[1], 'rank']
                effect_size_B = self._rank_biserial_effect_size(data1, data2)
                ci_B = self._bootstrap_ci(data1, data2, self._rank_biserial_effect_size)
            posthoc_A, posthoc_B = None, None
            if p_A < self.alpha and df[factorA].nunique() > 2:
                posthoc_A = self._posthoc_mannwhitney(df, 'rank', factorA)
            if p_B < self.alpha and df[factorB].nunique() > 2:
                posthoc_B = self._posthoc_mannwhitney(df, 'rank', factorB)
            self.raw_results = {
                "main_effect_A": {
                    "name": factorA,
                    "F": f_obs_A,
                    "p": p_A,
                    "df_num": df_num_A,
                    "df_den": df_den_A,
                    "effect_size": effect_size_A,
                    "ci_lower": ci_A[0],
                    "ci_upper": ci_A[1],
                    "posthoc_tests": posthoc_A
                },
                "main_effect_B": {
                    "name": factorB,
                    "F": f_obs_B,
                    "p": p_B,
                    "df_num": df_num_B,
                    "df_den": df_den_B,
                    "effect_size": effect_size_B,
                    "ci_lower": ci_B[0],
                    "ci_upper": ci_B[1],
                    "posthoc_tests": posthoc_B
                },
                "interaction": {
                    "name": interaction,
                    "F": f_obs_interaction,
                    "p": p_interaction,
                    "df_num": df_num_interaction,
                    "df_den": df_den_interaction,
                    "effect_size": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "posthoc_tests": None
                }
            }
            self.results = self._standardize_results(self.raw_results)
            
            # Store BOTH original and ranked data for Excel export transparency
            self.results["ranked_data"] = {}
            self.results["original_data"] = {}
            self.results["aggregated_data"] = {}
            
            valid_groups_A = df[factorA].unique()
            valid_groups_B = df[factorB].unique()
            
            # Store data for Factor A
            for group_name in valid_groups_A:
                # Original data (before aggregation)
                orig_data = original_df[original_df[factorA] == group_name][dv].tolist()
                self.results["original_data"][f"{factorA}_{group_name}"] = orig_data
                
                # Aggregated data (means used for ranking)
                agg_data = df[df[factorA] == group_name][dv].tolist()
                self.results["aggregated_data"][f"{factorA}_{group_name}"] = agg_data
                
                # Ranked data (used for test)
                rank_data = df[df[factorA] == group_name]['rank'].tolist()
                self.results["ranked_data"][f"{factorA}_{group_name}"] = rank_data
            
            # Store data for Factor B
            for group_name in valid_groups_B:
                orig_data = original_df[original_df[factorB] == group_name][dv].tolist()
                self.results["original_data"][f"{factorB}_{group_name}"] = orig_data
                
                agg_data = df[df[factorB] == group_name][dv].tolist()
                self.results["aggregated_data"][f"{factorB}_{group_name}"] = agg_data
                
                rank_data = df[df[factorB] == group_name]['rank'].tolist()
                self.results["ranked_data"][f"{factorB}_{group_name}"] = rank_data
            
            # Store descriptive statistics for both original and ranked data
            import numpy as np
            from scipy import stats as scipy_stats
            
            self.results["descriptive"] = {}
            self.results["descriptive_ranked"] = {}
            
            # Descriptives for Factor A
            for group_name in valid_groups_A:
                orig_data = original_df[original_df[factorA] == group_name][dv].tolist()
                if orig_data:
                    n = len(orig_data)
                    mean_val = np.mean(orig_data)
                    std_val = np.std(orig_data, ddof=1) if n > 1 else 0
                    se_val = std_val / np.sqrt(n) if n > 0 else 0
                    
                    if n > 1:
                        ci_margin = scipy_stats.t.ppf(0.975, n-1) * se_val
                        ci_lower = mean_val - ci_margin
                        ci_upper = mean_val + ci_margin
                    else:
                        ci_lower = ci_upper = mean_val
                    
                    self.results["descriptive"][f"{factorA}_{group_name}"] = {
                        "n": n,
                        "mean": mean_val,
                        "median": np.median(orig_data),
                        "std": std_val,
                        "stderr": se_val,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "min": np.min(orig_data),
                        "max": np.max(orig_data)
                    }
                
                # Rank data descriptives
                rank_data = df[df[factorA] == group_name]['rank'].tolist()
                if rank_data:
                    self.results["descriptive_ranked"][f"{factorA}_{group_name}"] = {
                        "n": len(rank_data),
                        "mean_rank": np.mean(rank_data),
                        "median_rank": np.median(rank_data),
                        "std_rank": np.std(rank_data, ddof=1) if len(rank_data) > 1 else 0,
                        "min_rank": np.min(rank_data),
                        "max_rank": np.max(rank_data)
                    }
            
            # Descriptives for Factor B
            for group_name in valid_groups_B:
                orig_data = original_df[original_df[factorB] == group_name][dv].tolist()
                if orig_data:
                    n = len(orig_data)
                    mean_val = np.mean(orig_data)
                    std_val = np.std(orig_data, ddof=1) if n > 1 else 0
                    se_val = std_val / np.sqrt(n) if n > 0 else 0
                    
                    if n > 1:
                        ci_margin = scipy_stats.t.ppf(0.975, n-1) * se_val
                        ci_lower = mean_val - ci_margin
                        ci_upper = mean_val + ci_margin
                    else:
                        ci_lower = ci_upper = mean_val
                    
                    self.results["descriptive"][f"{factorB}_{group_name}"] = {
                        "n": n,
                        "mean": mean_val,
                        "median": np.median(orig_data),
                        "std": std_val,
                        "stderr": se_val,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "min": np.min(orig_data),
                        "max": np.max(orig_data)
                    }
                
                # Rank data descriptives
                rank_data = df[df[factorB] == group_name]['rank'].tolist()
                if rank_data:
                    self.results["descriptive_ranked"][f"{factorB}_{group_name}"] = {
                        "n": len(rank_data),
                        "mean_rank": np.mean(rank_data),
                        "median_rank": np.median(rank_data),
                        "std_rank": np.std(rank_data, ddof=1) if len(rank_data) > 1 else 0,
                        "min_rank": np.min(rank_data),
                        "max_rank": np.max(rank_data)
                    }
            
            # Store explanation for transparency
            self.results["data_explanation"] = {
                "transformation": "Rank transformation (global ranking)",
                "aggregation": f"Replicates averaged before ranking (from {original_df.shape[0]} to {df.shape[0]} observations)",
                "ranking_method": "Global ranking across all aggregated observations",
                "test_performed_on": "Ranked data using Freedman-Lane permutation scheme"
            }
            
            return self.results
        except Exception as e:
            results = {
                "test_type": "non-parametric",
                "test_name": self.test_name,
                "error": str(e),
                "effects": []
            }
            return results


class NonParametricRMANOVA(NonParametricANOVA):
    """Non-parametric Repeated Measures ANOVA using rank transformation + permutation test."""
    
    def __init__(self, n_perm=1000, random_state=None, alpha=0.05):
        super().__init__(n_perm, random_state, alpha)
        self.test_name = "Non-parametric RM-ANOVA (Rank + Permutation)"
    
    def run(self, df, dv, subject, within):
        """
        Run non-parametric RM-ANOVA using Freedman–Lane permutation for the within effect.
        We first collapse any replicate measurements for each (subject, time-point) by taking the mean.
        """
        try:
            within_factor = within[0]
            
            # Store original data before aggregation for Excel export
            original_df = df.copy()
            
            # ======= AGGREGATE TRIPLICATES/REPLICATES =======
            print(f"DEBUG: Original data shape: {df.shape}")
            counts = df.groupby([subject, within_factor])[dv].count().reset_index(name="n_obs")
            max_obs = counts["n_obs"].max()
            print(f"DEBUG: Max observations per cell: {max_obs}")
            
            if max_obs > 1:
                print(f"DEBUG: Found replicates - aggregating by mean...")
                df = df.groupby([subject, within_factor], as_index=False)[dv].mean()
                print(f"DEBUG: After aggregation shape: {df.shape}")
            else:
                print("DEBUG: No replicates found, proceeding with original data")
            # ================================================
            
            # Now that each (subject, time) cell is a single number, proceed with global ranking
            print("DEBUG: Using global ranking across all observations")
            df['rank'] = df[dv].rank(method='average')
            print(f"DEBUG: Rank distribution: {df['rank'].describe()}")
            
            try:
                observed_aov = pg.rm_anova(data=df, dv='rank', within=within_factor, subject=subject, detailed=True)
                f_obs = float(observed_aov.loc[observed_aov['Source']==within_factor, 'F'])
                df_num = int(observed_aov.loc[observed_aov['Source']==within_factor, 'DF'])
                df_den = int(observed_aov.loc[observed_aov['Source']=='Error', 'DF'])
                print(f"DEBUG: Observed F={f_obs:.4f}, df=({df_num},{df_den})")
            except Exception as obs_e:
                print(f"Error in observed ANOVA: {str(obs_e)}")
                f_obs, df_num, df_den = 0, 1, 1
            
            # Call Freedman-Lane with proper error handling
            try:
                f_obs, f_perms = self._freedman_lane_permutation(df, dv, [], within_factor, self.n_perm, 'rm_anova', subject=subject, within=within)
                print(f"DEBUG: Freedman-Lane returned F={f_obs:.4f}")
            except Exception as fl_e:
                print(f"Error in Freedman-Lane permutation: {str(fl_e)}")
                f_perms = np.zeros(self.n_perm)
                # Set f_obs to a safe value if it wasn't set before
                if 'f_obs' not in locals() or not np.isfinite(f_obs):
                    f_obs = 0
            
            # Continue with post-processing
            p_val = (1 + np.sum(f_perms >= f_obs)) / (1 + self.n_perm)
            print(f"DEBUG: Calculated p-value: {p_val:.4f}")
            
            effect_size, ci = None, (None, None)
            posthoc = None
            
            # Post-hoc tests for more than 2 groups OR significant result with 2 groups
            if df[within_factor].nunique() == 2:
                print("DEBUG: Two groups detected - performing Wilcoxon test")
                levels = df[within_factor].unique()
                wide = df.pivot(index=subject, columns=within_factor, values='rank')
                data1 = wide[levels[0]].dropna()
                data2 = wide[levels[1]].dropna()
                try:
                    stat, wilcoxon_p = stats.wilcoxon(data1, data2)
                    std_dev = (data1 - data2).std()
                    effect_size = (data1 - data2).mean() / std_dev if std_dev and std_dev != 0 else None
                    ci = self._bootstrap_ci(data1, data2, lambda x, y: (x - y).mean())
                    
                    # Create a pairwise comparison entry
                    posthoc = [{
                        "level1": str(levels[0]),
                        "level2": str(levels[1]),
                        "statistic": float(stat),
                        "p_raw": float(wilcoxon_p),
                        "p_adj": float(wilcoxon_p),  # No correction needed for single comparison
                        "significant": wilcoxon_p < self.alpha,
                        "adjustment": "none"
                    }]
                    print(f"DEBUG: Wilcoxon test result: p={wilcoxon_p:.4f}")
                except Exception as wilcox_e:
                    print(f"DEBUG: Wilcoxon test failed: {str(wilcox_e)}")
                    effect_size = None
                    ci = (None, None)
                    posthoc = None
                    
            elif df[within_factor].nunique() > 2:
                print(f"DEBUG: {df[within_factor].nunique()} groups detected - performing post-hoc tests")
                posthoc = self._posthoc_wilcoxon(df, subject, within_factor)
                if posthoc:
                    print(f"DEBUG: Post-hoc tests completed: {len(posthoc)} comparisons")
                else:
                    print("DEBUG: Post-hoc tests returned no results")
                
            self.raw_results = {
                "within_effect": {
                    "name": within_factor,
                    "F": f_obs,
                    "p": p_val,
                    "df_num": df_num,
                    "df_den": df_den,
                    "effect_size": effect_size,
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                    "posthoc_tests": posthoc
                }
            }
            
            print(f"DEBUG: Raw results created with posthoc_tests: {posthoc is not None}")
            
            self.results = self._standardize_results(self.raw_results)
            
            # DEBUG: Check if pairwise comparisons were properly created
            if "effects" in self.results and len(self.results["effects"]) > 0:
                effect = self.results["effects"][0]
                if "posthoc_tests" in effect and effect["posthoc_tests"]:
                    print(f"DEBUG: Standardized results contain {len(effect['posthoc_tests'])} post-hoc tests")
                else:
                    print("DEBUG: No post-hoc tests in standardized results")
            
            # Store BOTH original and ranked data for Excel export transparency
            self.results["ranked_data"] = {}
            self.results["original_data"] = {}
            self.results["aggregated_data"] = {}  # Store the aggregated means before ranking
            valid_groups = df[within_factor].unique()
            
            for group_name in valid_groups:
                # 1. Original data (all replicates before aggregation)
                original_group_data = original_df[original_df[within_factor] == group_name][dv].tolist()
                self.results["original_data"][group_name] = original_group_data
                
                # 2. Aggregated data (means used for ranking)
                aggregated_group_data = df[df[within_factor] == group_name][dv].tolist()
                self.results["aggregated_data"][group_name] = aggregated_group_data
                
                # 3. Ranked data (what was actually used for the test)
                rank_group_data = df[df[within_factor] == group_name]['rank'].tolist()
                self.results["ranked_data"][group_name] = rank_group_data
            
            # Store descriptive statistics for both original and ranked data
            import numpy as np
            from scipy import stats as scipy_stats
            
            self.results["descriptive"] = {}
            self.results["descriptive_ranked"] = {}
            
            for group_name in valid_groups:
                # Original data descriptives (before aggregation)
                orig_data = original_df[original_df[within_factor] == group_name][dv].tolist()
                if orig_data:
                    n = len(orig_data)
                    mean_val = np.mean(orig_data)
                    std_val = np.std(orig_data, ddof=1) if n > 1 else 0
                    se_val = std_val / np.sqrt(n) if n > 0 else 0
                    
                    # 95% CI for mean
                    if n > 1:
                        ci_margin = scipy_stats.t.ppf(0.975, n-1) * se_val
                        ci_lower = mean_val - ci_margin
                        ci_upper = mean_val + ci_margin
                    else:
                        ci_lower = ci_upper = mean_val
                    
                    self.results["descriptive"][group_name] = {
                        "n": n,
                        "mean": mean_val,
                        "median": np.median(orig_data),
                        "std": std_val,
                        "stderr": se_val,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "min": np.min(orig_data),
                        "max": np.max(orig_data)
                    }
                
                # Rank data descriptives (what was actually used for the test)
                rank_data = df[df[within_factor] == group_name]['rank'].tolist()
                if rank_data:
                    n_ranks = len(rank_data)
                    mean_rank = np.mean(rank_data)
                    std_rank = np.std(rank_data, ddof=1) if n_ranks > 1 else 0
                    
                    self.results["descriptive_ranked"][group_name] = {
                        "n": n_ranks,
                        "mean_rank": mean_rank,
                        "median_rank": np.median(rank_data),
                        "std_rank": std_rank,
                        "min_rank": np.min(rank_data),
                        "max_rank": np.max(rank_data)
                    }
            
            # Store explanation for transparency
            self.results["data_explanation"] = {
                "transformation": "Rank transformation (global ranking)",
                "aggregation": f"Triplicates averaged before ranking (from {original_df.shape[0]} to {df.shape[0]} observations)",
                "ranking_method": "Global ranking across all aggregated observations",
                "test_performed_on": "Ranked data using Freedman-Lane permutation scheme"
            }
            
            return self.results
        except Exception as e:
            # Create and return the error results in one step
            return {
                "test_type": "non-parametric",
                "test_name": self.test_name,
                "error": str(e),
                "effects": []
            }

    def _posthoc_wilcoxon(self, df, subject, within_factor):
        """Pairwise Wilcoxon signed-rank tests with Holm-Sidak correction."""
        levels = df[within_factor].unique()
        pairs = list(combinations(levels, 2))
        results = []
        pvals = []
        wide = df.pivot(index=subject, columns=within_factor, values='rank')
        
        for a, b in pairs:
            data1 = wide[a].dropna()
            data2 = wide[b].dropna()
            # Only keep subjects with both values
            mask = data1.index.intersection(data2.index)
            if len(mask) < 2:
                stat, p = np.nan, np.nan
            else:
                stat, p = stats.wilcoxon(data1.loc[mask], data2.loc[mask])
            pvals.append(p)
            results.append({"level1": a, "level2": b, "statistic": stat, "p_raw": p})
            
        # Always use Holm-Sidak for adjustment (not Holm)
        valid_pvals = np.array([p for p in pvals if p is not None and not np.isnan(p)])
        if len(valid_pvals) > 0:
            valid_indices = [i for i, p in enumerate(pvals) if p is not None and not np.isnan(p)]
            reject, p_adj = multipletests(valid_pvals, alpha=self.alpha, method='holm-sidak')[:2]
            
            # Set adjusted p-values and significance flags
            for i, r in enumerate(results):
                if i in valid_indices:
                    idx = valid_indices.index(i)
                    r["p_adj"] = p_adj[idx]
                    r["significant"] = reject[idx]
                else:
                    r["p_adj"] = 1.0
                    r["significant"] = False
                r["adjustment"] = "holm-sidak"  # Specify the exact method
        
        return results


class NonParametricMixedANOVA(NonParametricANOVA):
    """Non-parametric Mixed ANOVA using rank transformation + permutation test."""
    
    def __init__(self, n_perm=1000, random_state=None, alpha=0.05):
        super().__init__(n_perm, random_state, alpha)
        self.test_name = "Non-parametric Mixed ANOVA (Rank + Permutation)"
    
    def run(self, df, dv, subject, between, within):
        """
        Run non-parametric Mixed ANOVA using Freedman–Lane permutation for each effect.
        """
        try:
            between_factor = between[0]
            within_factor = within[0]
            
            # Store original data before aggregation for Excel export
            original_df = df.copy()
            
            # ======= AGGREGATE TRIPLICATES/REPLICATES =======
            print(f"DEBUG: Original data shape: {df.shape}")
            counts = df.groupby([subject, between_factor, within_factor])[dv].count().reset_index(name="n_obs")
            max_obs = counts["n_obs"].max()
            print(f"DEBUG: Max observations per cell: {max_obs}")
            
            if max_obs > 1:
                print(f"DEBUG: Found replicates - aggregating by mean...")
                df = df.groupby([subject, between_factor, within_factor], as_index=False)[dv].mean()
                print(f"DEBUG: After aggregation shape: {df.shape}")
            else:
                print("DEBUG: No replicates found, proceeding with original data")
            # ================================================
            
            interaction = f"{within_factor} * {between_factor}"
            df['rank'] = df.groupby(subject)[dv].transform(lambda x: x.rank(method='average'))
            observed_aov = pg.mixed_anova(data=df, dv='rank', between=between_factor, within=within_factor, subject=subject, detailed=True)
            f_obs_between = float(observed_aov.loc[observed_aov['Source']==between_factor, 'F'])
            f_obs_within = float(observed_aov.loc[observed_aov['Source']==within_factor, 'F'])
            f_obs_inter = float(observed_aov.loc[observed_aov['Source']==interaction, 'F'])
            df_num_between = int(observed_aov.loc[observed_aov['Source']==between_factor, 'ddof1'])
            df_den_between = int(observed_aov.loc[observed_aov['Source']==between_factor, 'ddof2'])
            df_num_within = int(observed_aov.loc[observed_aov['Source']==within_factor, 'ddof1'])
            df_den_within = int(observed_aov.loc[observed_aov['Source']==within_factor, 'ddof2'])
            df_num_inter = int(observed_aov.loc[observed_aov['Source']==interaction, 'ddof1'])
            df_den_inter = int(observed_aov.loc[observed_aov['Source']==interaction, 'ddof2'])
            f_obs_between, f_between = self._freedman_lane_permutation(df, dv, [between_factor], between_factor, self.n_perm, 'mixed_anova', subject=subject, within=within)
            f_obs_within, f_within = self._freedman_lane_permutation(df, dv, [between_factor], within_factor, self.n_perm, 'mixed_anova', subject=subject, within=within)
            f_obs_inter, f_inter = self._freedman_lane_permutation(df, dv, [between_factor], interaction, self.n_perm, 'mixed_anova', subject=subject, within=within)
            p_between = (1 + np.sum(f_between >= f_obs_between)) / (1 + self.n_perm)
            p_within = (1 + np.sum(f_within >= f_obs_within)) / (1 + self.n_perm)
            p_inter = (1 + np.sum(f_inter >= f_obs_inter)) / (1 + self.n_perm)
            effect_size_between, ci_between = None, (None, None)
            effect_size_within, ci_within = None, (None, None)
            posthoc_between, posthoc_within = None, None
            if df[between_factor].nunique() == 2:
                levels = df[between_factor].unique()
                data1 = df.loc[df[between_factor]==levels[0], 'rank']
                data2 = df.loc[df[between_factor]==levels[1], 'rank']
                effect_size_between = self._rank_biserial_effect_size(data1, data2)
                ci_between = self._bootstrap_ci(data1, data2, self._rank_biserial_effect_size)
            elif p_between < self.alpha and df[between_factor].nunique() > 2:
                posthoc_between = self._posthoc_mannwhitney(df, 'rank', between_factor)
            if df[within_factor].nunique() == 2:
                levels = df[within_factor].unique()
                wide = df.pivot(index=subject, columns=within_factor, values='rank')
                data1 = wide[levels[0]].dropna()
                data2 = wide[levels[1]].dropna()
                stat, _ = stats.wilcoxon(data1, data2)
                std_dev = (data1 - data2).std()
                effect_size_within = (data1 - data2).mean() / std_dev if std_dev and std_dev != 0 else None
                ci_within = self._bootstrap_ci(data1, data2, lambda x, y: (x - y).mean())
            elif p_within < self.alpha and df[within_factor].nunique() > 2:
                posthoc_within = self._posthoc_wilcoxon(df, subject, within_factor)
            self.raw_results = {
                "between_effect": {
                    "name": between_factor,
                    "F": f_obs_between,
                    "p": p_between,
                    "df_num": df_num_between,
                    "df_den": df_den_between,
                    "effect_size": effect_size_between,
                    "ci_lower": ci_between[0],
                    "ci_upper": ci_between[1],
                    "posthoc_tests": posthoc_between
                },
                "within_effect": {
                    "name": within_factor,
                    "F": f_obs_within,
                    "p": p_within,
                    "df_num": df_num_within,
                    "df_den": df_den_within,
                    "effect_size": effect_size_within,
                    "ci_lower": ci_within[0],
                    "ci_upper": ci_within[1],
                    "posthoc_tests": posthoc_within
                },
                "interaction": {
                    "name": interaction,
                    "F": f_obs_inter,
                    "p": p_inter,
                    "df_num": df_num_inter,
                    "df_den": df_den_inter,
                    "effect_size": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "posthoc_tests": None
                }
            }
            self.results = self._standardize_results(self.raw_results)
            
            # Store BOTH original and ranked data for Excel export transparency
            self.results["ranked_data"] = {}
            self.results["original_data"] = {}
            self.results["aggregated_data"] = {}
            
            valid_groups_between = df[between_factor].unique()
            valid_groups_within = df[within_factor].unique()
            
            # Store data for Between Factor
            for group_name in valid_groups_between:
                orig_data = original_df[original_df[between_factor] == group_name][dv].tolist()
                self.results["original_data"][f"{between_factor}_{group_name}"] = orig_data
                
                agg_data = df[df[between_factor] == group_name][dv].tolist()
                self.results["aggregated_data"][f"{between_factor}_{group_name}"] = agg_data
                
                rank_data = df[df[between_factor] == group_name]['rank'].tolist()
                self.results["ranked_data"][f"{between_factor}_{group_name}"] = rank_data
            
            # Store data for Within Factor
            for group_name in valid_groups_within:
                orig_data = original_df[original_df[within_factor] == group_name][dv].tolist()
                self.results["original_data"][f"{within_factor}_{group_name}"] = orig_data
                
                agg_data = df[df[within_factor] == group_name][dv].tolist()
                self.results["aggregated_data"][f"{within_factor}_{group_name}"] = agg_data
                
                rank_data = df[df[within_factor] == group_name]['rank'].tolist()
                self.results["ranked_data"][f"{within_factor}_{group_name}"] = rank_data
            
            # Store descriptive statistics for both original and ranked data
            import numpy as np
            from scipy import stats as scipy_stats
            
            self.results["descriptive"] = {}
            self.results["descriptive_ranked"] = {}
            
            # Descriptives for Between Factor
            for group_name in valid_groups_between:
                orig_data = original_df[original_df[between_factor] == group_name][dv].tolist()
                if orig_data:
                    n = len(orig_data)
                    mean_val = np.mean(orig_data)
                    std_val = np.std(orig_data, ddof=1) if n > 1 else 0
                    se_val = std_val / np.sqrt(n) if n > 0 else 0
                    
                    if n > 1:
                        ci_margin = scipy_stats.t.ppf(0.975, n-1) * se_val
                        ci_lower = mean_val - ci_margin
                        ci_upper = mean_val + ci_margin
                    else:
                        ci_lower = ci_upper = mean_val
                    
                    self.results["descriptive"][f"{between_factor}_{group_name}"] = {
                        "n": n,
                        "mean": mean_val,
                        "median": np.median(orig_data),
                        "std": std_val,
                        "stderr": se_val,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "min": np.min(orig_data),
                        "max": np.max(orig_data)
                    }
                
                rank_data = df[df[between_factor] == group_name]['rank'].tolist()
                if rank_data:
                    self.results["descriptive_ranked"][f"{between_factor}_{group_name}"] = {
                        "n": len(rank_data),
                        "mean_rank": np.mean(rank_data),
                        "median_rank": np.median(rank_data),
                        "std_rank": np.std(rank_data, ddof=1) if len(rank_data) > 1 else 0,
                        "min_rank": np.min(rank_data),
                        "max_rank": np.max(rank_data)
                    }
            
            # Descriptives for Within Factor  
            for group_name in valid_groups_within:
                orig_data = original_df[original_df[within_factor] == group_name][dv].tolist()
                if orig_data:
                    n = len(orig_data)
                    mean_val = np.mean(orig_data)
                    std_val = np.std(orig_data, ddof=1) if n > 1 else 0
                    se_val = std_val / np.sqrt(n) if n > 0 else 0
                    
                    if n > 1:
                        ci_margin = scipy_stats.t.ppf(0.975, n-1) * se_val
                        ci_lower = mean_val - ci_margin
                        ci_upper = mean_val + ci_margin
                    else:
                        ci_lower = ci_upper = mean_val
                    
                    self.results["descriptive"][f"{within_factor}_{group_name}"] = {
                        "n": n,
                        "mean": mean_val,
                        "median": np.median(orig_data),
                        "std": std_val,
                        "stderr": se_val,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "min": np.min(orig_data),
                        "max": np.max(orig_data)
                    }
                
                rank_data = df[df[within_factor] == group_name]['rank'].tolist()
                if rank_data:
                    self.results["descriptive_ranked"][f"{within_factor}_{group_name}"] = {
                        "n": len(rank_data),
                        "mean_rank": np.mean(rank_data),
                        "median_rank": np.median(rank_data),
                        "std_rank": np.std(rank_data, ddof=1) if len(rank_data) > 1 else 0,
                        "min_rank": np.min(rank_data),
                        "max_rank": np.max(rank_data)
                    }
            
            # Store explanation for transparency
            self.results["data_explanation"] = {
                "transformation": "Rank transformation (within-subject ranking)",
                "aggregation": f"Replicates averaged before ranking (from {original_df.shape[0]} to {df.shape[0]} observations)",
                "ranking_method": "Within-subject ranking for mixed design",
                "test_performed_on": "Ranked data using Freedman-Lane permutation scheme"
            }
            
            return self.results
        except Exception as e:
            results = {
                "test_type": "non-parametric",
                "test_name": self.test_name,
                "error": str(e),
                "effects": []
            }
            return results

class NonParametricFactory:
    """Factory class to create the appropriate non-parametric ANOVA test based on design."""
    
    @staticmethod
    def create_nonparametric_test(design_type, n_perm=1000, random_state=None, alpha=0.05):
        """
        Create appropriate non-parametric test based on design type.
        
        Parameters:
        -----------
        design_type : str
            Type of design: 'two_way', 'rm_anova', or 'mixed_anova'
        n_perm : int
            Number of permutations
        random_state : int or None
            Random seed
        alpha : float
            Significance level
            
        Returns:
        --------
        NonParametricANOVA
            Instance of appropriate non-parametric test class with Holm-Sidak correction for post-hoc tests
        """
        if design_type == 'two_way':
            return NonParametricTwoWayANOVA(n_perm, random_state, alpha)
        elif design_type == 'rm_anova':
            return NonParametricRMANOVA(n_perm, random_state, alpha)
        elif design_type == 'mixed_anova':
            return NonParametricMixedANOVA(n_perm, random_state, alpha)
        else:
            raise ValueError(f"Unknown design type: {design_type}")
        
    @staticmethod
    def create_test(test_type):
        """Backward compatibility method for older code."""
        if test_type == 'two_way_anova':
            return NonParametricFactory.create_nonparametric_test('two_way')
        elif test_type == 'repeated_measures_anova':
            return NonParametricFactory.create_nonparametric_test('rm_anova')
        elif test_type == 'mixed_anova':
            return NonParametricFactory.create_nonparametric_test('mixed_anova')
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    @staticmethod
    def _standardize_results(results):
        """
        Standardizes results format for consistent Excel export.
        Ensures all required keys exist with appropriate default values.
        """
        # If this is already a standardized result (has flat structure), return as-is
        if "test" in results and "p_value" in results:
            return results
        
        # If this looks like a non-parametric result, standardize it
        if "effects" in results or "permutation_test" in results:
            try:
                from nonparametricanovas import NonParametricFactory
                return NonParametricFactory.standardize_results_for_export(results)
            except ImportError:
                print("Warning: NonParametricFactory not available for result standardization")
        
        # Ensure basic structure exists for parametric results
        standardized = {
            "test": results.get("test", "Unknown test"),
            "test_type": results.get("test_type", results.get("recommendation", "parametric")),
            "p_value": results.get("p_value"),
            "statistic": results.get("statistic"),
            "df1": results.get("df1"),
            "df2": results.get("df2"),
            "effect_size": results.get("effect_size"),
            "effect_size_type": results.get("effect_size_type"),
            "confidence_interval": results.get("confidence_interval"),
            "power": results.get("power"),
            "alpha": results.get("alpha", 0.05),
            "pairwise_comparisons": results.get("pairwise_comparisons", []),
            "descriptive": results.get("descriptive", {}),
            "raw_data": results.get("raw_data", {}),
            "groups": results.get("groups", []),
            "normality_tests": results.get("normality_tests", {}),
            "variance_test": results.get("variance_test", {}),
            "transformation": results.get("transformation", "None"),
            "factors": results.get("factors", []),
            "interactions": results.get("interactions", [])
        }
        
        # Copy over any additional keys that might be present
        for key, value in results.items():
            if key not in standardized:
                standardized[key] = value
        
        return standardized
    
    @staticmethod
    def standardize_results_for_export(results):
        """
        Konvertiert verschachtelte nicht-parametrische Ergebnisse
        in ein flaches Format für den Excel-Export.
        """
        if not results:
            return {}

        flat = {
            # Basisinfo
            "test": results.get("test_name", "Non-parametric ANOVA"),
            "test_type": "non-parametric",
            "permutation_test": True,
            "permutation_scheme": "Freedman-Lane",
            # Platzhalter, wird unten gefüllt:
            "statistic": None,
            "p_value": None,
            "df1": None,
            "df2": None,
            "effect_size": None,
            "effect_size_type": None,
            "confidence_interval": (None, None),
            # Gruppendaten (wird unten ergänzt)
            "raw_data": results.get("original_data", {}),
            "ranked_data": results.get("ranked_data", {}),
            "aggregated_data": results.get("aggregated_data", {}),
            "descriptive": results.get("descriptive", {}),
            # Faktoren und Interaktionen
            "factors": [],
            "interactions": [],
            # Post-Hoc
            "pairwise_comparisons": [],
            # Gruppenliste
            "groups": []
        }

        # 1) Fülle statistic, p_value, df1, df2, effect_size, confidence_interval
        main_eff = None
        for eff in results.get("effects", []):
            nm = eff.get("name", "")
            # Nicht-Interaktionen (kein '*' und kein ':')
            if nm and "*" not in nm and ":" not in nm:
                main_eff = eff
                break
        if not main_eff and results.get("effects"):
            main_eff = results["effects"][0]

        if main_eff:
            flat["statistic"] = main_eff.get("F")
            flat["p_value"]  = main_eff.get("p")
            flat["df1"]      = main_eff.get("df_num")
            flat["df2"]      = main_eff.get("df_den")
            flat["effect_size"] = main_eff.get("effect_size")
            # Besser: tatsächlichen Typ übernehmen, statt pauschal „eta²“
            flat["effect_size_type"] = eff.get("effect_size_type", None) or "r"
            flat["confidence_interval"] = (
                main_eff.get("ci_lower"),
                main_eff.get("ci_upper")
            )

        # 2) Fülle Faktoren und Interaktionen
        for eff in results.get("effects", []):
            nm = eff.get("name", "")
            if "*" in nm or ":" in nm:
                flat["interactions"].append(nm)
            else:
                flat["factors"].append(nm)

            # 3) Post-Hoc-Tests aus eff["posthoc_tests"] extrahieren
            for post in eff.get("posthoc_tests", []) or []:
                # Prüfen, ob paired (Wilcoxon) oder unpaired (Mann-Whitney) sein soll
                testname = post.get("test", None)
                if not testname:
                    # Falls du per Konvention „paired“ kennzeichnest:
                    if post.get("paired", False):
                        testname = "Wilcoxon signed-rank"
                    else:
                        testname = "Mann-Whitney U"

                flat["pairwise_comparisons"].append({
                    "group1": post.get("level1", ""),
                    "group2": post.get("level2", ""),
                    "test": testname,
                    "p_value": post.get("p_adj", post.get("p_raw")),
                    "statistic": post.get("statistic"),
                    "significant": post.get("significant", False),
                    "corrected": True,
                    "correction_method": post.get("adjustment", "holm-sidak"),
                    "effect_size": post.get("effect_size", None),
                    "effect_size_type": post.get("effect_size_type", None),
                    "confidence_interval": post.get("confidence_interval", (None, None))
                })

        # 4) Gruppenliste befüllen (Basis: original_data keys)
        for grp in results.get("original_data", {}):
            flat["groups"].append(grp)

        return flat