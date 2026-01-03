---
layout: default
title: "Ziyad BENKHADAJ"
---

<!-- Image of Ziyad -->
<div class="profile-image">
  <img src="{{ site.baseurl }}/assets/photo_Ziyad_Alpes.JPG" alt="Ziyad BENKHADAJ" style="width: 200px; height: auto;">
</div>

<!-- English Content -->
<div id="content-en">
  <h1>Welcome to my website</h1>
  <p>This is my personal website where you can find various information about me and the services I offer.</p>
  
  <ul>
    <li>CV</li>
    <li>Philosophical opinions</li>
    <li>Math and physics tutoring, from high school to higher education (preparatory classes, university, medical school...)</li>
  </ul>

  <h2>Contact</h2>
  <p>Email: <a href="mailto:ziyad.benkhadaj@gmail.com">ziyad.benkhadaj@gmail.com</a></p>
</div>

<!-- Real-time date and time -->
<div id="datetime" style="font-size: 18px; font-family: Arial, sans-serif; color: white;"></div>

<script>
  function updateRealTime() {
    var currentDate = new Date();
    var options = { 
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', 
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true 
    };
    document.getElementById('datetime').textContent = currentDate.toLocaleDateString('en-US', options);
  }

  setInterval(updateRealTime, 1000);
  updateRealTime();
</script>
