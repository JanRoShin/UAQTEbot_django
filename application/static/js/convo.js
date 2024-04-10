function sendMessage() {
  var userInput = document.getElementById("user-input").value;
  if (!userInput.trim()) {
    return; // Exit the function if input is empty
  }
  displayMessage("You", userInput);
  document.getElementById("user-input").value = "";
  fetch("/chat/", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      "X-CSRFToken": getCookie("csrftoken"), // Get CSRF token
    },
    body: "message=" + encodeURIComponent(userInput),
  })
    .then((response) => response.json())
    .then((data) => {
      displayMessage("UAQTEbot", data.message);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

// Add an event listener to the input field to detect Enter key press
document
  .getElementById("user-input")
  .addEventListener("keyup", function (event) {
    // Check if the Enter key (key code 13) is pressed
    if (event.keyCode === 13) {
      // Prevent the default action of the Enter key (submitting the form)
      event.preventDefault();
      // Call the sendMessage function to send the message
      sendMessage();
    }
  });

function displayMessage(sender, message) {
  var chatContainer = document.getElementById("chat-container");
  var messageElement = document.createElement("div");
  messageElement.classList.add("message");
  if (sender === "You") {
    messageElement.classList.add("user");
  } else {
    messageElement.classList.add("UAQTEbot");
  }
  messageElement.innerText = sender + ": " + message;
  chatContainer.appendChild(messageElement);
  chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to bottom
}

// Function to get CSRF token from cookies
function getCookie(name) {
  var cookieValue = null;
  if (document.cookie && document.cookie !== "") {
    var cookies = document.cookie.split(";");
    for (var i = 0; i < cookies.length; i++) {
      var cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === name + "=") {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}
