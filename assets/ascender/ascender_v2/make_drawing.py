#!/usr/bin/env python3
"""Dimensioned schematic of the adapter + BOM -> drawing.png / drawing.svg (run after rebuilding adapter.stl)."""
import trimesh, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
a=trimesh.load('adapter.stl'); w=trimesh.load('../../robots/g1/_menagerie/unitree_g1/assets/right_wrist_yaw_link.STL'); w.apply_scale(1000)
OR='#b0592d'
def draw(ax, mesh, normal, origin, idx, color, lw=0.6):
    s=mesh.section(plane_origin=origin, plane_normal=normal)
    if s is None: return
    for e in s.entities: p=s.vertices[e.points][:,idx]; ax.plot(p[:,0],p[:,1],color=color,lw=lw)
def silhouette(ax, mesh, idx, color):
    v=mesh.vertices[:,idx]; f=mesh.faces; seg=np.concatenate([v[f[:,[0,1]]],v[f[:,[1,2]]],v[f[:,[2,0]]]])
    ax.add_collection(LineCollection(seg,colors=color,linewidths=0.15,alpha=0.5))
def dim(ax, p1, p2, text, off=(0,0), color='k'):
    p1=np.array(p1)+off; p2=np.array(p2)+off; ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle='<->',mutation_scale=8,color=color,lw=0.8))
    ax.text(*((p1+p2)/2+np.array([0,1.5])),text,ha='center',va='bottom',fontsize=7,color=color)
fig,axs=plt.subplots(2,2,figsize=(16,12)); fig.suptitle('Ascender adapter v2 — Unitree G1 right_wrist_yaw_link   |   adapter.scad (OpenSCAD+BOSL2)   |   mm, wrist link frame (X toward hand, Z up)',fontsize=11)
ax=axs[0,0]; ax.set_title('SIDE — section y = −3 (plug axis)'); silhouette(ax,w,[0,2],'#999'); silhouette(ax,a,[0,2],OR); draw(ax,a,[0,1,0],[0,-3,0],[0,2],'k',0.9)
dim(ax,(16.5,22),(41.5,22),'plug 25'); dim(ax,(41.5,40),(47.5,40),'flange 6'); dim(ax,(30,-40),(41.5,-40),'collar 11.5')
ax.annotate('4× M3 ×10 (stock clamp axis, Z)',xy=(33.5,-35),xytext=(-5,-70),fontsize=7,arrowprops=dict(arrowstyle='->',lw=0.6))
ax.annotate('Ø12 shoulder bolt + Ø18/12 sleeve\nin the Petzl attachment eye',xy=(67.7,-37.9),xytext=(70,-90),fontsize=7,arrowprops=dict(arrowstyle='->',lw=0.6))
ax.annotate('wrist shell (Unitree STL)',xy=(20,28),xytext=(-15,45),fontsize=7,color='#666',arrowprops=dict(arrowstyle='->',lw=0.6,color='#666'))
ax=axs[0,1]; ax.set_title('FRONT — view from +X (hand side)'); silhouette(ax,a,[1,2],OR); draw(ax,a,[1,0,0],[44,0,0],[1,2],'k',0.9); draw(ax,a,[1,0,0],[34,0,0],[1,2],'k',0.6)
dim(ax,(-22.3,0),(16.3,0),'plug Ø38.6 (opening Ø38.95)',off=(0,-3)); dim(ax,(-39,-45),(33,-45),'flange 72 wide')
ax.annotate('split gap 1.0',xy=(-36,0),xytext=(-60,15),fontsize=7,arrowprops=dict(arrowstyle='->',lw=0.6))
ax=axs[1,0]; ax.set_title('TOP — view from +Z'); silhouette(ax,w,[0,1],'#999'); silhouette(ax,a,[0,1],OR); draw(ax,a,[0,0,1],[0,0,-25],[0,1],'k',0.9)
dim(ax,(30,-42),(41.5,-42),'collar 11.5'); ax.annotate('M3 holes 2 per side\n(x = 33.5, 37.5)',xy=(35.5,-36),xytext=(0,-60),fontsize=7,arrowprops=dict(arrowstyle='->',lw=0.6))
ax=axs[1,1]; ax.axis('off')
bom=[('1','Adapter body','1','PA12-CF (SLS/FDM) 71 g — or 6061-T6, 190 g','adapter.scad / adapter.step'),
     ('2','M3 × 10 socket head','4','A2 stainless','collar clamp, along Z (same as Unitree clamp screws)'),
     ('3','M3 nyloc nut','4','A2 stainless','—'),
     ('4','Ø12 shoulder bolt M10, L=25','1','12.9 steel','pin through cheeks + eye'),
     ('5','Sleeve Ø18 / Ø12 × 6','1','6061 or POM','fills the Petzl Ø18 eye'),
     ('6','Petzl Basic ascender (B18)','1','—','scanned part; untouched'),
     ('7','Unitree wrist clamp + 2× M3','1','stock','retained; retention of the plug = TODO')]
ax.text(0,0.98,'BOM',fontsize=11,weight='bold',va='top'); y=0.92
for row in [('#','Part','Qty','Material','Note')]+bom:
    for x,txt in zip([0,0.05,0.42,0.5,0.75],row): ax.text(x,y,txt,fontsize=7.5,va='top',weight='bold' if row[0]=='#' else 'normal')
    y-=0.06
ax.text(0,0.4,'Interfaces measured on Unitree right_wrist_yaw_link.STL (see wrist_interface_drawing.png):\n  end wall x 39.5–41.5, opening Ø38.95 @ (y −3, z 0), D-outline y −30..+23.8, z −30.5..+30\n'
        'Petzl eye (scan): tool (x −9, y +4.9, z 19.4), Ø18; plate 5.8 mm (y +2.0..+7.8) → cheek pocket 6.2\n'
        'Tool pose: USD pose + 12.5 mm X (clears the flange). Mesh check: 0 overlap with the Petzl, cam untouched\n'
        'Loads: 350 N rope → 18 N·m on the wrist (actuators 5 N·m) → forearm aligned with the rope during climb\n'
        'Colour: Petzl orange #b0592d (sampled from the scan basecolor)\n'
        'OPEN: plug retention inside the wrist socket (needs a real G1 or a Unitree drawing)',fontsize=7.5,va='top',family='monospace')
for ax in axs.ravel()[:3]: ax.set_aspect('equal'); ax.grid(True,alpha=.25); ax.autoscale(); ax.margins(0.15)
plt.tight_layout(); plt.savefig('drawing.png',dpi=130); plt.savefig('drawing.svg'); print('drawing ok')
