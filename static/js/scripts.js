// scripts.js

// Wait for the document to fully load
document.addEventListener("DOMContentLoaded", function () {
    // Navbar - Highlight active links
    const navbarLinks = document.querySelectorAll(".navbar-nav .nav-link");
    navbarLinks.forEach(link => {
        if (link.href === window.location.href) {
            link.classList.add("active");
        }
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener("click", function (e) {
            e.preventDefault();
            const targetId = this.getAttribute("href").substring(1);
            document.getElementById(targetId)?.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        });
    });

    // Form Validation on Submit
    const form = document.querySelector(".form-detail");
    if (form) {
        form.addEventListener("submit", function (event) {
            event.preventDefault();
            const isValid = validateForm(form);
            if (isValid) {
                showLoading(event.submitter); // Show loading animation
                form.submit(); // Proceed with form submission if valid
            }
        });
    }

    // Hover effect for social media icons in the footer
    const socialIcons = document.querySelectorAll(".footer .fa-stack");
    socialIcons.forEach(icon => {
        icon.addEventListener("mouseover", () => icon.style.color = "#ffffff");
        icon.addEventListener("mouseout", () => icon.style.color = "#ccc");
    });
});

// Validate the form fields before submitting
function validateForm(form) {
    const fields = form.querySelectorAll(".input-text");
    let isValid = true;

    fields.forEach(field => {
        if (!field.value.trim()) {
            showValidationError(field, "This field is required.");
            isValid = false;
        } else if (isNaN(field.value)) {
            showValidationError(field, "Please enter a valid number.");
            isValid = false;
        } else {
            clearValidationError(field);
        }
    });

    return isValid;
}

// Show error message below the input field
function showValidationError(field, message) {
    clearValidationError(field);

    const error = document.createElement("small");
    error.className = "validation-error";
    error.style.color = "red";
    error.textContent = message;
    field.parentElement.appendChild(error);
}

// Clear any existing error messages
function clearValidationError(field) {
    const existingError = field.parentElement.querySelector(".validation-error");
    if (existingError) {
        existingError.remove();
    }
}

// Display loading animation on form submit
function showLoading(button) {
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
}

// Handle submission feedback for training or prediction
function handleSubmission(event, buttonId) {
    event.preventDefault();
    const button = document.getElementById(buttonId);
    if (button) {
        showLoading(button);
    }
    // Optional: AJAX submission can be added here to handle without page reload
}

// AJAX for dynamic prediction results display (Optional)
async function submitPredictionForm(event) {
    event.preventDefault();
    const formData = new FormData(event.target);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });
        const result = await response.json();

        if (response.ok) {
            displayPredictionResult(result.prediction);
        } else {
            displayError("Prediction failed. Please try again.");
        }
    } catch (error) {
        displayError("An error occurred. Please try again later.");
    }
}

// Display prediction results
function displayPredictionResult(result) {
    const resultContainer = document.querySelector(".prediction-box");
    if (resultContainer) {
        resultContainer.textContent = `Prediction: ${result}`;
        resultContainer.classList.add("show");
    }
}

// Display error message for predictions
function displayError(message) {
    const resultContainer = document.querySelector(".prediction-box");
    if (resultContainer) {
        resultContainer.textContent = message;
        resultContainer.classList.add("error");
    }
}
