function [linelength]=computeLengthOfTriangleWithRotation(keyz,triangle,rots)

%%z lies on one edge of triangle
if keyz == triangle(1,3) & keyz == triangle(2,3)
    linelength = getTwoPointsLength(triangle(1,1:2),triangle(2,1:2));
    return;
elseif keyz == triangle(2,3) & keyz == triangle(3,3)
    linelength = getTwoPointsLength(triangle(2,1:2),triangle(3,1:2));
    return;
elseif keyz == triangle(3,3) & keyz == triangle(1,3)
    linelength = getTwoPointsLength(triangle(3,1:2),triangle(1,1:2));
    return;
end

intersection = 0;
for currentindex = 1:3
    %%z crosses one vertex 
    if keyz == triangle(currentindex,3)
        p0index = mod(currentindex,3)+1;
        p1index = mod(p0index,3)+1;
        zz = (keyz - triangle(p0index,3)) / (triangle(p1index,3)-triangle(p0index,3));
        %%z crosses one vertex and one edge of triangle
        if zz>0 & zz<1
            xx = triangle(currentindex,1) ;
            yy = triangle(currentindex,2) ;
            xx1 = triangle(p0index,1) + zz*(triangle(p1index,1)-triangle(p0index,1));
            yy1 = triangle(p0index,2) + zz*(triangle(p1index,2)-triangle(p0index,2));
            linelength = sqrt((xx1-xx)^2 + (yy1-yy)^2);
            return;
        %%z just cross one point
        else
            linelength = 0;
            return;
        end
        
    end
    
    %%z crosses two edges of triangle
    %%zz = z-z0 / z1-z0
    zz = (keyz - triangle(currentindex,3)) / (triangle(mod(currentindex,3)+1,3)-triangle(currentindex,3));
    if zz>0 & zz<1
        %%x = x0 + zz*(x1 - x0)
        nextindex = mod(currentindex,3)+1;
        xx = triangle(currentindex,1) + zz*(triangle(nextindex,1)-triangle(currentindex,1));
        yy = triangle(currentindex,2) + zz*(triangle(nextindex,2)-triangle(currentindex,2));
        if intersection == 0
            xx1 = xx;
            yy1 = yy;
            intersection = 1;
            continue;
            %end intersection == 0
        elseif intersection == 1
            xx2 = xx;
            yy2 = yy;
            linelength = sqrt((xx1-xx2)^2 + (yy1-yy2)^2);
            %linep = [xx yy keyz; xx1 yy1 keyz];
            %linep = linep*rots;
            %triangle = triangle*rots;
            %line(triangle(:,1),triangle(:,2),triangle(:,3),'Color','black','LineStyle','--');
            %line(linep(:,1),linep(:,2),linep(:,3),'Color','red','LineStyle','--');
            %linelength
            break;
        end %end intersection == 1
    end %zz>0 && zz<1
    linelength=0;
end %currentindex = 1:3
%break;
end




