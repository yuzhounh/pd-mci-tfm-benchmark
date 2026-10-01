# Step 18: Nomogram of the final model (7-predictor LR with bootstrap uniform shrinkage).
#
# Adapted from the parent project's step_16_nomogram.R. The nomogram uses the shrunk
# coefficients and re-estimated intercept from step 14 (step_14_shrunk_coefficients.csv),
# i.e. exactly the model reported in the paper. rms draws the nomogram from an lrm object, so
# the model is set up with lrm on the training data and its coefficients are replaced by the
# shrunk ones; only the coefficients and the predictor distributions (datadist) are used for
# drawing. Statistics printed by print(fit), validate() or calibrate() would refer to R's own
# fit and must not be cited.
#
# Check: predicted probabilities from the replaced coefficients must equal the step 14
# shrunk-model predictions for the 500 test participants.
#
# Outputs: step_18_nomogram.pdf and step_18_nomogram_points.txt (points per predictor value).

suppressPackageStartupMessages(library(rms))

train <- read.csv("PPMI_9_train.csv")
test <- read.csv("PPMI_9_test.csv")
coefs <- read.csv("step_14_shrunk_coefficients.csv", check.names = FALSE)
beta <- setNames(coefs$beta_shrunk, coefs$Variable)

prepare <- function(d) {
    out <- data.frame(
        Sex = factor(d$SEX, levels = c(0, 1), labels = c("Female", "Male")),
        Education = d$EDUCYRS,
        FH_PD = factor(d$fampd_bin, levels = c(0, 1), labels = c("No", "Yes")),
        Other_symptom = factor(d$sym_other, levels = c(0, 1), labels = c("No", "Yes")),
        UPSIT_percentile = d$upsit_pctl,
        GDS = d$gds,
        Caudate_SBR = d$MIA_CAUDATE_mean
    )
    label(out$Sex) <- "Sex"
    label(out$Education) <- "Education (years)"
    label(out$FH_PD) <- "Family history of PD"
    label(out$Other_symptom) <- "Other initial symptom"
    label(out$UPSIT_percentile) <- "UPSIT percentile"
    label(out$GDS) <- "GDS score"
    label(out$Caudate_SBR) <- "Caudate SBR (DAT)"
    out
}
train_df <- prepare(train)
train_df$MCI <- train$MCI
test_df <- prepare(test)
stopifnot(!anyNA(train_df), !anyNA(test_df))

dd <- datadist(train_df)
options(datadist = "dd")
fit <- lrm(MCI ~ Sex + Education + FH_PD + Other_symptom + UPSIT_percentile + GDS + Caudate_SBR,
           data = train_df, x = TRUE, y = TRUE)

# lrm coefficient name -> step 14 variable name
name_map <- c(
    "Intercept" = "Intercept",
    "Sex=Male" = "SEX",
    "Education" = "EDUCYRS",
    "FH_PD=Yes" = "fampd_bin",
    "Other_symptom=Yes" = "sym_other",
    "UPSIT_percentile" = "upsit_pctl",
    "GDS" = "gds",
    "Caudate_SBR" = "MIA_CAUDATE_mean"
)
stopifnot(setequal(names(fit$coefficients), names(name_map)), setequal(unname(name_map), names(beta)))
fit$coefficients[] <- beta[name_map[names(fit$coefficients)]]
stopifnot(!anyNA(fit$coefficients))

# Check against the step 14 shrunk-model predictions for the test participants
p_r <- predict(fit, newdata = test_df, type = "fitted")
p_step14 <- read.csv("step_14_shrunk_predictions.csv")$y_prob
max_diff <- max(abs(p_r - p_step14))
cat(sprintf("Max |p(nomogram model) - p(step 14 shrunk model)| over %d test participants: %.2e\n",
            length(p_r), max_diff))
if (max_diff > 1e-10) stop("Nomogram coefficients do not reproduce the step 14 shrunk model")

nomo <- nomogram(fit,
    Education = seq(4, 20, by = 4),
    UPSIT_percentile = seq(0, 100, by = 20),
    GDS = seq(0, 15, by = 3),
    Caudate_SBR = seq(0.2, 1.8, by = 0.4),
    fun = plogis,
    fun.at = c(0.05, seq(0.1, 0.7, 0.1)),
    funlabel = "Probability of PD-MCI"
)

sink("step_18_nomogram_points.txt")
cat("Nomogram of the final model (shrunk LR); points per predictor value\n\n")
print(nomo)
sink()

pdf_path <- "step_18_nomogram.pdf"
pdf(pdf_path, width = 11, height = 7)
par(lwd = 1.3)
plot(nomo, xfrac = 0.28, cex.var = 1.1, cex.axis = 0.95)
dev.off()
if (Sys.which("pdfcrop") != "") {
    system2("pdfcrop", c(pdf_path, pdf_path))
}
cat("Saved step_18_nomogram.pdf and step_18_nomogram_points.txt\n")
