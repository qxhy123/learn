(function(){
  function flatten(){ return (window.TUTORIAL||[]).flatMap(function(group){ return (group.chapters || group.items || []).map(function(item){ return Object.assign({part: group.part}, item); }); }); }
  function normalizedPath(){ return location.pathname.replace(/\\/g,'/').replace(/^.*\/html\//,'').replace(/^\/+/, ''); }
  function currentItem(items){
    var path = normalizedPath();
    return items.find(function(item){ return item.path === path; }) || items.find(function(item){ return path.endsWith(item.path); }) || null;
  }
  function link(item, label, cls){
    if(!item) return '<span class="navlink disabled">.</span>';
    return '<a class="navlink '+(cls||'')+'" href="'+relToRoot(item.path)+'">'+label+'<span>'+item.title+'</span></a>';
  }
  function relToRoot(target){
    var here = normalizedPath();
    var depth = Math.max(0, here.split('/').length - 1);
    return '../'.repeat(depth) + target;
  }
  function render(){
    var items = flatten();
    var cur = currentItem(items);
    if(!cur) return;
    var idx = items.findIndex(function(item){ return item.id === cur.id; });
    var prev = items[idx-1];
    var next = items[idx+1];
    var up = normalizedPath() === 'index.html' ? null : {title:'目录', path:'index.html'};
    document.querySelectorAll('.topnav').forEach(function(el){ el.innerHTML = link(prev,'← ','prev') + (up ? link(up,'↑ ','nav-up') : '<span></span>') + link(next,'→ ','next'); });
    document.querySelectorAll('.bottomnav').forEach(function(el){ el.innerHTML = link(prev,'← ','prev') + (up ? link(up,'↑ ','nav-up') : '<span></span>') + link(next,'→ ','next'); });
    var frame = document.getElementById('sidebar');
    if(frame && frame.contentWindow){ frame.addEventListener('load', function(){ frame.contentWindow.postMessage({type:'current-chapter', id:cur.id}, '*'); }); }
  }
  document.addEventListener('DOMContentLoaded', render);
})();
