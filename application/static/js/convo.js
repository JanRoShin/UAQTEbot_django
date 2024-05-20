function sendMessage() {
  var userInput = document.getElementById("user-input").value;
  if (!userInput.trim()) {
    return; // Exit the function if input is empty
  }
  displayMessage("You", userInput);
  document.getElementById("user-input").value = "";
  fetch("/chat/?user_input=" + encodeURIComponent(userInput))
    .then((response) => response.json())
    .then((data) => {
      displayMessage("UAQTEbot", data.bot_response);
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
  messageElement.classList.add(sender);
  messageElement.innerText = sender + ":\n\n     " + message;
  chatContainer.appendChild(messageElement);
  chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to bottom
}
