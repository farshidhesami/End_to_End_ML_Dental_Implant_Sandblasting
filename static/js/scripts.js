// scripts.js

// Wait for the document to fully load
document.addEventListener("DOMContentLoaded", function () {
    // Highlight active links in the navbar
    const navbarLinks = document.querySelectorAll(".navbar-nav .nav-link");
    navbarLinks.forEach(link => {
        if (link.href === window.location.href) {
            link.classList.add("active");
        }
    });

    // Enable smooth scrolling for anchor links
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

    // Form validation on submit
    const form = document.querySelector(".form-detail");
    if (form) {
        form.addEventListener("submit", function (event) {
            event.preventDefault();
            const isValid = validateForm(form);
            if (isValid) {
                showLoading(event.submitter); // Show loading spinner
                form.submit(); // Submit the form
            }
        });
    }

    // Hover effect for social media icons
    const socialIcons = document.querySelectorAll(".footer .fa-stack");
    socialIcons.forEach(icon => {
        icon.addEventListener("mouseover", () => icon.style.color = "#ffffff");
        icon.addEventListener("mouseout", () => icon.style.color = "#ccc");
    });
});

// Validate form fields before submission
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

// Display validation error message
function showValidationError(field, message) {
    clearValidationError(field);

    const error = document.createElement("small");
    error.className = "validation-error";
    error.style.color = "red";
    error.textContent = message;
    field.parentElement.appendChild(error);
}

// Clear existing validation error messages
function clearValidationError(field) {
    const existingError = field.parentElement.querySelector(".validation-error");
    if (existingError) {
        existingError.remove();
    }
}

// Show loading spinner on submit button
function showLoading(button) {
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
}

// Optional: Handle AJAX submission dynamically
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

// Display prediction result dynamically
function displayPredictionResult(result) {
    const resultContainer = document.querySelector(".prediction-box");
    if (resultContainer) {
        resultContainer.textContent = `Prediction: ${result}`;
        resultContainer.classList.add("show");
    }
}

// Display error message
function displayError(message) {
    const resultContainer = document.querySelector(".prediction-box");
    if (resultContainer) {
        resultContainer.textContent = message;
        resultContainer.classList.add("error");
    }
}
